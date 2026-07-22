"""compute/00b_full_graph.py → results/full_graph.json + results/embeddings.npy
Full GNN graph: spring layout on patent citation+litigation graph.
Stores all 1331 patents + litigation edges + test plaintiff list.
"""
import sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import load_context, save_json, RESULTS_DIR

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False

try:
    from sklearn.decomposition import PCA
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


PAT_PAT_TYPES = [
    ("patent", "cites",    "patent"),
    ("patent", "cited_by", "patent"),
    ("patent", "sued",     "patent"),
    ("patent", "sued_by",  "patent"),
]


def build_full_graph(ctx: dict) -> None:
    print("[00b_full_graph] Building full graph layout...")
    h_emb      = ctx["h_emb"]       # (N, 64)
    pat_list   = ctx["pat_list"]
    data       = ctx["data"]
    state      = ctx["state"]
    test_edges = state["test_edges"]  # (2, N_test) tensor
    df_asgn    = ctx["df_asgn"]

    N = len(pat_list)
    W, H = 1400, 1000

    # ── 1. Build networkx graph from patent-patent edges ─────────────────────
    print(f"    Building patent graph ({N} nodes)...")
    G = nx.Graph() if _HAS_NX else None

    citation_pairs   = []
    litigation_pairs = []

    for et in PAT_PAT_TYPES:
        if et not in data.edge_index_dict:
            continue
        ei  = data.edge_index_dict[et].cpu().numpy()
        rel = et[1]
        is_lit = rel in ("sued", "sued_by")
        for s, d in zip(ei[0], ei[1]):
            s, d = int(s), int(d)
            if s >= N or d >= N:
                continue
            if is_lit:
                litigation_pairs.append((s, d))
            else:
                citation_pairs.append((s, d))
            if G is not None:
                G.add_edge(f"pat_{s}", f"pat_{d}")

    print(f"    citation pairs={len(citation_pairs)}, litigation pairs={len(litigation_pairs)}")

    # ── 2. Layout: spring on citation+litigation graph ────────────────────────
    if G is not None and G.number_of_edges() > 0:
        print("    Computing spring layout (may take ~30s)...")
        if G is not None:
            for i in range(N):
                G.add_node(f"pat_{i}")
        pos = nx.spring_layout(G, k=2.5, seed=42, iterations=60)
        pos_arr = np.array([[pos[f"pat_{i}"][0], pos[f"pat_{i}"][1]] for i in range(N)])
    elif _HAS_SKLEARN:
        print("    Falling back to PCA layout...")
        pca     = PCA(n_components=2, random_state=42)
        pos_arr = pca.fit_transform(h_emb)
    else:
        print("    Falling back to SVD layout...")
        centered = h_emb - h_emb.mean(0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        pos_arr  = centered @ Vt[:2].T

    # Scale to pixel range
    for col in range(2):
        mn, mx = pos_arr[:, col].min(), pos_arr[:, col].max()
        pos_arr[:, col] = (pos_arr[:, col] - mn) / (mx - mn + 1e-8) * (W if col == 0 else H) - (W if col == 0 else H) / 2

    # ── 3. Company lookup + color index ──────────────────────────────────────
    def get_owner(label):
        if df_asgn is None:
            return ""
        try:
            col  = df_asgn.columns[0]
            rows = df_asgn[df_asgn[col] == label]
            return str(rows.iloc[0, 1]) if len(rows) else ""
        except Exception:
            return ""

    pat_owner = [get_owner(pat_list[i]) for i in range(N)]
    co_counts = Counter(o for o in pat_owner if o)
    top_cos   = [co for co, _ in co_counts.most_common(12)]
    co_to_idx = {co: i for i, co in enumerate(top_cos)}

    # ── 4. Build node records ─────────────────────────────────────────────────
    nodes = []
    for i in range(N):
        label = pat_list[i]
        owner = pat_owner[i]
        nodes.append({
            "id":          f"pat_{i}",
            "label":       label,
            "short_label": label[:14],
            "x":           float(pos_arr[i, 0]),
            "y":           float(pos_arr[i, 1]),
            "owner":       owner,
            "co_idx":      co_to_idx.get(owner, -1),
        })

    # ── 5. Litigation edges for display (de-duplicated) ───────────────────────
    lit_set  = set()
    lit_display = []
    for s, d in litigation_pairs:
        key = (min(s, d), max(s, d))
        if key not in lit_set:
            lit_set.add(key)
            lit_display.append({"s": s, "d": d})
    print(f"    Unique litigation edges for display: {len(lit_display)}")

    # ── 6. Test plaintiff list ────────────────────────────────────────────────
    te_np  = test_edges.cpu().numpy() if hasattr(test_edges, "cpu") else np.array(test_edges)
    N_test = te_np.shape[1]

    gt_by_src = defaultdict(list)
    for pi in range(N_test):
        s = int(te_np[0, pi])
        d = int(te_np[1, pi])
        gt_by_src[s].append({"pair_idx": pi, "dst_idx": d, "gt_label": pat_list[d]})

    test_plaintiffs = []
    for s, gt_list in sorted(gt_by_src.items()):
        test_plaintiffs.append({
            "src_idx":  s,
            "id":       f"pat_{s}",
            "label":    pat_list[s],
            "owner":    pat_owner[s],
            "pair_idx": gt_list[0]["pair_idx"],
            "gts": [
                {"dst_idx": g["dst_idx"], "id": f"pat_{g['dst_idx']}", "gt_label": g["gt_label"]}
                for g in gt_list
            ],
        })

    print(f"    Patents={N}, Test plaintiffs={len(test_plaintiffs)}, Top cos={top_cos[:4]}")

    # ── 7. Save embeddings ────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(RESULTS_DIR / "embeddings.npy"), h_emb.astype(np.float32))
    print(f"    embeddings.npy  shape={h_emb.shape}")

    # ── 8. Save full_graph.json ───────────────────────────────────────────────
    save_json({
        "metadata": {
            "n_patents":           N,
            "n_test_plaintiffs":   len(test_plaintiffs),
            "layout":              "spring" if G is not None else "pca",
            "top_companies":       top_cos,
            "created_at":          ctx["created_at"],
        },
        "nodes":            nodes,
        "litigation_edges": lit_display,
        "test_plaintiffs":  test_plaintiffs,
    }, "full_graph.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--khop", type=int, default=2)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()
    ctx = load_context(args.pair, args.khop, args.topk)
    build_full_graph(ctx)
