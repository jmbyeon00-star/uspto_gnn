"""compute/02_attention.py → results/attention.json
Extract native GAT attention weights per layer using return_attention_weights=True.
Also patches graph.json edges with attention values.
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from common import load_context, save_json, RELATION_MAP, RESULTS_DIR

LAYER_HEADS = [8, 4, 4, 2]   # head count per layer


def extract_attention(model, data) -> dict:
    """
    Returns:
        attn[layer_name][et_key] = {(src_global, dst_global): alpha_mean}
    """
    model.eval()
    x_pat = torch.cat([data["patent"].struct, data["patent"].x], dim=1)
    x = {
        "patent":  model.pat_proj(x_pat),
        "company": F.leaky_relu(model.co_proj(data["company"].x)),
        "ipc":     model.ipc_emb.weight,
    }

    results = {}
    with torch.no_grad():
        for layer_idx in range(4):
            hconv = model.convs[layer_idx]
            layer_key = f"layer{layer_idx + 1}"
            layer_dict: dict = {}

            for et, gat in hconv.convs.items():
                if et not in data.edge_index_dict:
                    continue
                src_t, rel_t, dst_t = et
                ei   = data.edge_index_dict[et]
                x_in = x[src_t] if src_t == dst_t else (x[src_t], x[dst_t])

                _, (ei_out, alpha) = gat(x_in, ei, return_attention_weights=True)
                # alpha: (E, heads)
                a = alpha.abs().mean(dim=-1).cpu().numpy()   # (E,)
                srcs = ei_out[0].cpu().numpy()
                dsts = ei_out[1].cpu().numpy()

                edge_map: dict = {}
                for i in range(len(a)):
                    edge_map[(int(srcs[i]), int(dsts[i]))] = round(float(a[i]), 4)
                layer_dict[rel_t] = {"src_type": src_t, "dst_type": dst_t, "edges": edge_map}

            results[layer_key] = layer_dict

            # Forward to get next-layer inputs
            if layer_idx < 3:
                x_new = hconv(x, data.edge_index_dict)
                bn_d  = model.bns[layer_idx]
                act   = _CONV_ACTS[layer_idx]
                x = {
                    k: model.drop(act(bn_d[k](v.view(-1, model.hid)))).view(v.shape[0], -1)
                    for k, v in x_new.items()
                }

    return results


def build_attention(ctx: dict) -> None:
    print("[02_attention] Extracting GAT attention weights...")
    model    = ctx["model"]
    data     = ctx["data"]
    src_idx  = ctx["src_idx"]
    dst_idx  = ctx["dst_idx"]
    pat_list = ctx["pat_list"]

    # Load graph.json to get subgraph edge list and IDs
    graph_path = RESULTS_DIR / "graph.json"
    if not graph_path.exists():
        raise FileNotFoundError("Run 00_graph.py first.")
    with open(graph_path) as f:
        graph = json.load(f)
    edges = graph["edges"]

    # Parse node IDs → global indices
    def nid_to_idx(nid: str):
        t, i = nid.rsplit("_", 1)
        return t, int(i)

    # Extract full attention
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "viz_app"))
    from backend.model import _CONV_ACTS
    globals()["_CONV_ACTS"] = _CONV_ACTS

    attn_full = extract_attention(model, data)

    # Map edge records to per-layer attention
    # Unify: for each subgraph edge, look up attention from matching layer/rel
    layer_output: dict = {}
    for layer_key, layer_dict in attn_full.items():
        records = []
        for edge in edges:
            if edge["relation"] not in ("citation", "litigation"):
                records.append({"edge_id": edge["id"], "attention": 0.0, "normalized_attention": 0.0})
                continue

            src_t, s = nid_to_idx(edge["source"])
            dst_t, d = nid_to_idx(edge["target"])
            if src_t != "pat" or dst_t != "pat":
                records.append({"edge_id": edge["id"], "attention": 0.0, "normalized_attention": 0.0})
                continue

            raw_type = edge["raw_type"]
            attn_val = 0.0
            if raw_type in layer_dict:
                attn_val = layer_dict[raw_type]["edges"].get((s, d), 0.0)

            records.append({"edge_id": edge["id"], "attention": attn_val, "normalized_attention": 0.0})

        # Normalize within layer
        vals = [r["attention"] for r in records]
        max_v = max(vals) if vals else 1.0
        for r in records:
            r["normalized_attention"] = round(r["attention"] / (max_v + 1e-8), 4)

        layer_output[layer_key] = records

    # Compute per-layer average attention on pat-pat edges
    layer_avg = {}
    for lk, recs in layer_output.items():
        vals = [r["attention"] for r in recs if r["attention"] > 0]
        layer_avg[lk] = round(float(np.mean(vals)) if vals else 0.0, 4)

    # Patch graph.json edges with average attention across layers
    for edge in edges:
        eid = edge["id"]
        attn_per_layer = {lk: next((r["attention"] for r in recs if r["edge_id"] == eid), 0.0)
                          for lk, recs in layer_output.items()}
        vals = [v for v in attn_per_layer.values() if v > 0]
        edge["attention"] = round(float(np.mean(vals)) if vals else 0.0, 4)
        edge["attention_layers"] = {lk: round(v, 4) for lk, v in attn_per_layer.items()}

    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    print("    ✓ graph.json patched with attention")

    result = {
        "layer_avg": layer_avg,
        "layers": layer_output,
        "created_at": ctx["created_at"],
    }
    save_json(result, "attention.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--khop", type=int, default=2)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    ctx = load_context(args.pair, args.khop, args.topk)
    build_attention(ctx)
