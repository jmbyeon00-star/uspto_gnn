"""compute/03_explanation.py → results/explanation.json
GNNExplainer + PGExplainer + SubgraphX (greedy LOO) on patent subgraph.
"""
import sys, copy, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from common import load_context, save_json, RESULTS_DIR, sigmoid

from torch_geometric.nn import GATConv
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import Data


# ── Surrogate model for GNNExplainer ─────────────────────────────────────────
class PatentExplainModel(nn.Module):
    """Thin 2-layer GAT on frozen 64-d embeddings.
    Returns per-node representations; link prediction = dot(src, dst)."""
    def __init__(self, in_dim=64, hidden=32, heads=2, dropout=0.1):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads,
                             dropout=dropout, add_self_loops=False)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, concat=False,
                             dropout=dropout, add_self_loops=False)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)  # (N, hidden)


def _train_surrogate(model, x, ei, src_i, dst_i, neg_ratio=3, epochs=120, lr=3e-3):
    """Quick training of surrogate on the subgraph."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    n   = x.size(0)

    # Positive: (src, dst), Negatives: random pairs
    pos_label = torch.tensor([1.0])
    neg_idx = [i for i in range(n) if i != src_i and i != dst_i]

    for _ in range(epochs):
        opt.zero_grad()
        out = model(x, ei)
        pos_score = (out[src_i] * out[dst_i]).sum()
        loss = F.binary_cross_entropy_with_logits(pos_score.unsqueeze(0), pos_label)

        if neg_idx:
            k = min(neg_ratio, len(neg_idx))
            negs = np.random.choice(neg_idx, k, replace=False)
            for ni in negs:
                neg_s = (out[src_i] * out[ni]).sum()
                loss = loss + F.binary_cross_entropy_with_logits(
                    neg_s.unsqueeze(0), torch.tensor([0.0]))
            loss = loss / (1 + k)

        loss.backward()
        opt.step()

    model.eval()
    return model


def _build_subgraph(ctx, graph_edges):
    """Extract patent-only subgraph tensors from graph.json edges."""
    graph_path = RESULTS_DIR / "graph.json"
    with open(graph_path) as f:
        graph = json.load(f)

    # Patent node ID → local index
    pat_nodes = [n for n in graph["nodes"] if n["type"] == "patent"]
    global2local = {}
    local2node   = {}
    for li, n in enumerate(pat_nodes):
        gi = int(n["id"].split("_")[1])
        global2local[gi] = li
        local2node[li]   = n

    src_global = ctx["src_idx"]
    dst_global = ctx["dst_idx"]
    src_li = global2local.get(src_global)
    dst_li = global2local.get(dst_global)

    h_emb = ctx["h_emb"]
    x     = torch.tensor(np.stack([h_emb[int(n["id"].split("_")[1])] for n in pat_nodes]),
                         dtype=torch.float)

    # Build edge_index from citation / litigation edges between patents
    srcs, dsts = [], []
    eid_map = {}  # (local_s, local_d) → edge_id
    for e in graph_edges:
        src_t = e["source"].split("_")[0]
        dst_t = e["target"].split("_")[0]
        if src_t != "pat" or dst_t != "pat":
            continue
        sg = int(e["source"].split("_")[1])
        dg = int(e["target"].split("_")[1])
        sl = global2local.get(sg)
        dl = global2local.get(dg)
        if sl is None or dl is None:
            continue
        srcs.append(sl)
        dsts.append(dl)
        eid_map[(sl, dl)] = e["id"]

    if not srcs:
        return None
    ei = torch.tensor([srcs, dsts], dtype=torch.long)
    return {"x": x, "ei": ei, "src_li": src_li, "dst_li": dst_li,
            "eid_map": eid_map, "local2node": local2node,
            "n_local": len(pat_nodes), "pat_nodes": pat_nodes}


def run_gnn_explainer(sub, explain_model):
    """Run PyG GNNExplainer on the surrogate model."""
    x, ei     = sub["x"], sub["ei"]
    src_li    = sub["src_li"]
    dst_li    = sub["dst_li"]

    # Wrapper: returns logit for the (src, dst) pair as "node prediction"
    class EdgePredictor(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x, edge_index):
            out = self.base(x, edge_index)
            scores = (out * out[dst_li]).sum(dim=-1, keepdim=True)  # (N, 1)
            return scores

    wrap = EdgePredictor(explain_model)
    wrap.eval()

    explainer = Explainer(
        model=wrap,
        algorithm=GNNExplainer(epochs=200, lr=0.01),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
        model_config=dict(
            mode="binary_classification",
            task_level="node",
            return_type="raw",
        ),
    )
    try:
        exp = explainer(x=x, edge_index=ei, index=torch.tensor([src_li]))
    except Exception as ex:
        print(f"    GNNExplainer error: {ex}")
        return {}, {}

    node_mask = exp.node_mask.squeeze().cpu().numpy()
    edge_mask = exp.edge_mask.squeeze().cpu().numpy()

    node_imp = {sub["pat_nodes"][i]["id"]: round(float(node_mask[i]), 4)
                for i in range(len(sub["pat_nodes"]))}
    edge_imp = {}
    srcs = ei[0].numpy(); dsts = ei[1].numpy()
    for i, (s, d) in enumerate(zip(srcs, dsts)):
        eid = sub["eid_map"].get((int(s), int(d)))
        if eid:
            edge_imp[eid] = round(float(edge_mask[i]) if i < len(edge_mask) else 0.0, 4)

    return node_imp, edge_imp


def run_pg_explainer(sub, explain_model):
    """PGExplainer with CPU copy workaround."""
    try:
        from torch_geometric.explain import PGExplainer
        x, ei  = sub["x"].cpu(), sub["ei"].cpu()
        src_li = sub["src_li"]
        dst_li = sub["dst_li"]
        model_cpu = copy.deepcopy(explain_model).cpu().eval()

        class EdgePredCPU(nn.Module):
            def __init__(self, base):
                super().__init__(); self.base = base
            def forward(self, x, edge_index):
                out = self.base(x, edge_index)
                return (out * out[dst_li]).sum(dim=-1, keepdim=True)

        wrap = EdgePredCPU(model_cpu)
        pg = Explainer(
            model=wrap,
            algorithm=PGExplainer(epochs=100, lr=0.003),
            explanation_type="phenomenon",
            edge_mask_type="object",
            model_config=dict(mode="binary_classification",
                              task_level="node", return_type="raw"),
        )
        # PGExplainer requires training phase
        for ep in range(100):
            pg.algorithm.train(ep, wrap, x, edge_index=ei,
                               target=torch.tensor([1.0]),
                               index=torch.tensor([src_li]))
        exp = pg(x=x, edge_index=ei, target=torch.tensor([1.0]),
                 index=torch.tensor([src_li]))
        edge_mask = exp.edge_mask.squeeze().cpu().numpy()
        srcs = ei[0].numpy(); dsts = ei[1].numpy()
        edge_imp = {}
        for i, (s, d) in enumerate(zip(srcs, dsts)):
            eid = sub["eid_map"].get((int(s), int(d)))
            if eid:
                edge_imp[eid] = round(float(edge_mask[i]) if i < len(edge_mask) else 0.0, 4)
        return edge_imp
    except Exception as ex:
        print(f"    PGExplainer skipped: {ex}")
        return {}


def run_subgraph_x(sub, explain_model) -> dict:
    """Greedy Leave-One-Out node importance (SubgraphX approx)."""
    x, ei  = sub["x"], sub["ei"]
    src_li = sub["src_li"]
    dst_li = sub["dst_li"]
    n      = sub["n_local"]

    def predict(x_in):
        with torch.no_grad():
            out = explain_model(x_in, ei)
            return torch.sigmoid((out[src_li] * out[dst_li]).sum()).item()

    base_p = predict(x)
    node_imp = {}
    for i in range(n):
        if i == src_li or i == dst_li:
            node_imp[sub["pat_nodes"][i]["id"]] = 1.0
            continue
        x_masked = x.clone()
        x_masked[i] = 0.0
        p_masked = predict(x_masked)
        node_imp[sub["pat_nodes"][i]["id"]] = round(max(0.0, base_p - p_masked), 4)

    # Top coalition = top-5 nodes by importance
    sorted_nodes = sorted(node_imp.items(), key=lambda kv: -kv[1])
    coalition = [nid for nid, _ in sorted_nodes[:5]]
    return {"node_importance": node_imp, "coalition": coalition}


def build_explanation(ctx: dict) -> None:
    print("[03_explanation] Running GNNExplainer + PGExplainer + SubgraphX...")

    # Load graph.json
    graph_path = RESULTS_DIR / "graph.json"
    if not graph_path.exists():
        raise FileNotFoundError("Run 00_graph.py first.")
    with open(graph_path) as f:
        graph = json.load(f)

    sub = _build_subgraph(ctx, graph["edges"])
    if sub is None:
        print("    No patent-patent edges found; skipping explanation.")
        save_json({"gnn_explainer": {}, "pg_explainer": {}, "subgraph_x": {}}, "explanation.json")
        return

    # Build and train surrogate
    explain_model = PatentExplainModel(in_dim=64)
    explain_model = _train_surrogate(explain_model, sub["x"], sub["ei"],
                                     sub["src_li"], sub["dst_li"])

    # GNNExplainer
    gnn_node_imp, gnn_edge_imp = run_gnn_explainer(sub, explain_model)
    important_nodes = sorted(gnn_node_imp, key=lambda k: -gnn_node_imp[k])[:8]
    important_edges = sorted(gnn_edge_imp, key=lambda k: -gnn_edge_imp[k])[:10]

    # Normalize GNN edge importance → update graph.json
    max_e = max(gnn_edge_imp.values()) if gnn_edge_imp else 1.0
    for edge in graph["edges"]:
        edge["importance"] = round(gnn_edge_imp.get(edge["id"], 0.0) / (max_e + 1e-8), 4)
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    print("    ✓ graph.json patched with GNN edge importance")

    # PGExplainer
    pg_edge_imp = run_pg_explainer(sub, explain_model)

    # SubgraphX
    sgx = run_subgraph_x(sub, explain_model)

    # Attention summary from attention.json
    attn_path = RESULTS_DIR / "attention.json"
    attn_summary = {}
    if attn_path.exists():
        with open(attn_path) as f:
            attn_data = json.load(f)
        for lk, recs in attn_data.get("layers", {}).items():
            top = sorted(recs, key=lambda r: -r["attention"])[:5]
            attn_summary[lk] = [r["edge_id"] for r in top if r["attention"] > 0]

    result = {
        "gnn_explainer": {
            "node_importance":  gnn_node_imp,
            "edge_importance":  gnn_edge_imp,
            "important_nodes":  important_nodes,
            "important_edges":  important_edges,
        },
        "pg_explainer": {
            "edge_importance": pg_edge_imp,
            "important_edges": sorted(pg_edge_imp, key=lambda k: -pg_edge_imp[k])[:10],
        },
        "subgraph_x": sgx,
        "attention_summary": {"top_edges_by_layer": attn_summary},
        "created_at": ctx["created_at"],
    }
    save_json(result, "explanation.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--khop", type=int, default=2)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    ctx = load_context(args.pair, args.khop, args.topk)
    build_explanation(ctx)
