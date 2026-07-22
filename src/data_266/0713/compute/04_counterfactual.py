"""compute/04_counterfactual.py → results/counterfactual.json
Edge-type removal + per-edge Top-K removal counterfactual analysis.
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import load_context, save_json, RELATION_MAP, sigmoid, RESULTS_DIR

# Edge types grouped by unified relation
TYPE_TO_RELATION = {
    ("patent","cites","patent"):       "citation",
    ("patent","cited_by","patent"):    "citation",
    ("patent","owned_by","company"):   "owner",
    ("company","owns","patent"):       "owner",
    ("patent","has_ipc","ipc"):        "ipc",
    ("ipc","rev_has_ipc","patent"):    "ipc",
    ("patent","sued","patent"):        "litigation",
    ("patent","sued_by","patent"):     "litigation",
}


def _predict_with_mask(model, data, src_idx, dst_idx,
                       masked_relations=None, masked_et_keys=None):
    """Forward pass with specified edge types masked out; return P(sue)."""
    model.eval()
    # Build edge_index_dict copy with some edges removed
    new_ei = {}
    for et, ei in data.edge_index_dict.items():
        rel = TYPE_TO_RELATION.get(et)
        if masked_relations and rel in masked_relations:
            # Keep only edges not involving src or dst
            mask = ~((ei[0] == src_idx) | (ei[1] == src_idx) |
                     (ei[0] == dst_idx) | (ei[1] == dst_idx))
            new_ei[et] = ei[:, mask]
        elif masked_et_keys and et in masked_et_keys:
            mask = ~((ei[0] == src_idx) | (ei[1] == src_idx) |
                     (ei[0] == dst_idx) | (ei[1] == dst_idx))
            new_ei[et] = ei[:, mask]
        else:
            new_ei[et] = ei

    with torch.no_grad():
        # Patch data temporarily
        orig = data.edge_index_dict.copy() if hasattr(data.edge_index_dict, 'copy') else dict(data.edge_index_dict)
        data._edge_store_dict = {}
        for et, ei in new_ei.items():
            src_t, rel_t, dst_t = et
            data[src_t, rel_t, dst_t].edge_index = ei
        h_dict = model(data)
        h_emb  = h_dict["patent"].cpu().numpy()
        p = sigmoid(h_emb[src_idx] @ h_emb[dst_idx])
        # Restore
        for et, ei in orig.items():
            src_t, rel_t, dst_t = et
            data[src_t, rel_t, dst_t].edge_index = ei
    return p


def _remove_single_edge(model, data, src_idx, dst_idx, et_key, edge_col):
    """Remove one specific edge (edge_col index in et_key's edge_index) and predict."""
    model.eval()
    et       = et_key
    ei_orig  = data[et[0], et[1], et[2]].edge_index
    mask     = torch.ones(ei_orig.size(1), dtype=torch.bool)
    mask[edge_col] = False
    ei_new   = ei_orig[:, mask]

    data[et[0], et[1], et[2]].edge_index = ei_new
    with torch.no_grad():
        h_dict = model(data)
        h_emb  = h_dict["patent"].cpu().numpy()
        p = sigmoid(h_emb[src_idx] @ h_emb[dst_idx])
    data[et[0], et[1], et[2]].edge_index = ei_orig
    return p


def build_counterfactual(ctx: dict) -> None:
    print("[04_counterfactual] Running counterfactual analysis...")
    model   = ctx["model"]
    data    = ctx["data"]
    src_idx = ctx["src_idx"]
    dst_idx = ctx["dst_idx"]
    topk    = ctx["topk"]
    p_base  = ctx["p_base"]

    # Load graph.json for edge metadata
    graph_path = RESULTS_DIR / "graph.json"
    if not graph_path.exists():
        raise FileNotFoundError("Run 00_graph.py first.")
    with open(graph_path) as f:
        graph = json.load(f)
    graph_edges = graph["edges"]
    edge_by_id  = {e["id"]: e for e in graph_edges}

    # ── 1. By unified relation type ───────────────────────────────────────────
    by_type = {}
    for relation in ["citation", "owner", "ipc", "litigation"]:
        try:
            p_after = _predict_with_mask(model, data, src_idx, dst_idx,
                                         masked_relations={relation})
        except Exception as ex:
            print(f"    {relation}: error ({ex})")
            p_after = p_base
        delta = round(p_after - p_base, 4)
        # Count how many graph edges are in this relation
        cnt = sum(1 for e in graph_edges if e["relation"] == relation)
        by_type[relation] = {
            "removed_count": cnt,
            "before":  round(p_base, 4),
            "after":   round(p_after, 4),
            "delta":   delta,
        }
        print(f"    {relation:12s}: {p_base:.4f} → {p_after:.4f}  Δ={delta:+.4f}")

    # ── 2. Per-edge (top-K GNN edges) ─────────────────────────────────────────
    # Find edges in the full graph that connect src/dst neighborhood
    load_expln = RESULTS_DIR / "explanation.json"
    top_edge_ids = []
    if load_expln.exists():
        with open(load_expln) as f:
            expln = json.load(f)
        gnn_imp = expln.get("gnn_explainer", {}).get("edge_importance", {})
        top_edge_ids = sorted(gnn_imp, key=lambda k: -gnn_imp.get(k, 0))[:topk]

    # Map graph edge_id → (et_key, column in full edge_index)
    # Build reverse lookup from graph.json source/target
    by_edge = []
    used_ids = set()

    pat_list = ctx["pat_list"]
    co_list  = ctx["co_list"]
    ipc_list = ctx["ipc_list"]

    def label_of(nid):
        t, i = nid.rsplit("_", 1); i = int(i)
        if t == "pat":  return pat_list[i]
        if t == "co":   return co_list[i]
        if t == "ipc":  return ipc_list[i]
        return nid

    def find_et_col(et_key, sg, dg):
        if et_key not in data.edge_index_dict:
            return None
        ei = data.edge_index_dict[et_key].cpu().numpy()
        for col, (s, d) in enumerate(zip(ei[0], ei[1])):
            if int(s) == sg and int(d) == dg:
                return col
        return None

    # Map raw_type → et_key
    RAW_TO_ET = {et[1]: et for et in TYPE_TO_RELATION}

    for eid in (top_edge_ids or [e["id"] for e in graph_edges
                                  if e["relation"] in ("citation","litigation")][:topk]):
        edge = edge_by_id.get(eid)
        if edge is None or eid in used_ids:
            continue
        used_ids.add(eid)

        src_t = edge["source"].split("_")[0]
        dst_t = edge["target"].split("_")[0]
        if src_t != "pat" or dst_t != "pat":
            continue
        sg = int(edge["source"].split("_")[1])
        dg = int(edge["target"].split("_")[1])

        raw = edge["raw_type"]
        et_key = RAW_TO_ET.get(raw)
        if et_key is None:
            continue
        col = find_et_col(et_key, sg, dg)
        if col is None:
            continue

        try:
            p_after = _remove_single_edge(model, data, src_idx, dst_idx, et_key, col)
        except Exception as ex:
            print(f"    edge {eid}: error ({ex})")
            p_after = p_base
        delta = round(p_after - p_base, 4)

        by_edge.append({
            "rank":         len(by_edge) + 1,
            "edge_id":      eid,
            "source":       edge["source"],
            "target":       edge["target"],
            "source_label": label_of(edge["source"]),
            "target_label": label_of(edge["target"]),
            "relation":     edge["relation"],
            "before":       round(p_base, 4),
            "after":        round(p_after, 4),
            "delta":        delta,
            "gnn_importance": gnn_imp.get(eid, 0.0) if top_edge_ids else 0.0,
        })

    # Re-rank by |delta|
    by_edge.sort(key=lambda x: x["delta"])
    for i, e in enumerate(by_edge):
        e["rank"] = i + 1

    # Patch graph.json is_counterfactual flag
    cf_edge_ids = {e["edge_id"] for e in by_edge[:3]}
    for edge in graph_edges:
        edge["is_counterfactual"] = edge["id"] in cf_edge_ids
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    print("    ✓ graph.json patched with is_counterfactual")

    result = {
        "base_probability": round(p_base, 4),
        "by_type":  by_type,
        "by_edge":  by_edge,
        "created_at": ctx["created_at"],
    }
    save_json(result, "counterfactual.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--khop", type=int, default=2)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    ctx = load_context(args.pair, args.khop, args.topk)
    build_counterfactual(ctx)
