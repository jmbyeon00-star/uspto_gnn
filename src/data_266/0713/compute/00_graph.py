"""compute/00_graph.py → results/graph.json
Build k-hop subgraph with layout, hop distances, and node metadata.
"""
import sys, collections
from pathlib import Path
import numpy as np
import networkx as nx
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import load_context, save_json, RELATION_MAP, sigmoid, DATA_DIR

# Edge types grouped by node kind
PAT_PAT = [
    ("patent", "cites",    "patent"),
    ("patent", "cited_by", "patent"),
    ("patent", "sued",     "patent"),
    ("patent", "sued_by",  "patent"),
]
PAT_CO = [
    ("patent",  "owned_by", "company"),
    ("company", "owns",     "patent"),
]
PAT_IPC = [
    ("patent", "has_ipc",     "ipc"),
    ("ipc",    "rev_has_ipc", "patent"),
]


def _bfs(adj: dict, start: int, max_hop: int) -> dict:
    dist = {start: 0}
    q    = collections.deque([start])
    while q:
        node = q.popleft()
        if dist[node] >= max_hop:
            continue
        for nb in adj.get(node, []):
            if nb not in dist:
                dist[nb] = dist[node] + 1
                q.append(nb)
    return dist


def build_graph(ctx: dict) -> None:
    print("[00_graph] Building subgraph...")
    data     = ctx["data"]
    src_idx  = ctx["src_idx"]
    dst_idx  = ctx["dst_idx"]
    khop     = ctx["khop"]
    pat_list = ctx["pat_list"]
    co_list  = ctx["co_list"]
    ipc_list = ctx["ipc_list"]
    h_emb    = ctx["h_emb"]

    # ── 1. Build patent adjacency from all pat-pat edges ─────────────────────
    adj_fwd: dict = collections.defaultdict(set)
    adj_bwd: dict = collections.defaultdict(set)
    for et in PAT_PAT:
        if et not in data.edge_index_dict:
            continue
        ei = data.edge_index_dict[et].cpu().numpy()
        for s, d in zip(ei[0], ei[1]):
            adj_fwd[int(s)].add(int(d))
            adj_bwd[int(d)].add(int(s))

    # ── 2. K-hop BFS from src and dst ────────────────────────────────────────
    hop_src = _bfs({k: list(v | adj_bwd.get(k, set())) for k, v in adj_fwd.items()},
                   src_idx, khop)
    hop_dst = _bfs({k: list(v | adj_bwd.get(k, set())) for k, v in adj_fwd.items()},
                   dst_idx, khop)

    subset_pat = set(hop_src) | set(hop_dst)
    subset_pat.add(src_idx)
    subset_pat.add(dst_idx)

    # ── 3. Find associated companies and IPCs ─────────────────────────────────
    co_of_pat:  dict = collections.defaultdict(set)   # pat_idx → {co_idx}
    ipc_of_pat: dict = collections.defaultdict(set)   # pat_idx → {ipc_idx}

    for et in PAT_CO:
        if et not in data.edge_index_dict:
            continue
        ei = data.edge_index_dict[et].cpu().numpy()
        st, dt = et[0], et[2]
        for s, d in zip(ei[0], ei[1]):
            if st == "patent" and dt == "company" and int(s) in subset_pat:
                co_of_pat[int(s)].add(int(d))
            elif st == "company" and dt == "patent" and int(d) in subset_pat:
                co_of_pat[int(d)].add(int(s))

    for et in PAT_IPC:
        if et not in data.edge_index_dict:
            continue
        ei = data.edge_index_dict[et].cpu().numpy()
        st, dt = et[0], et[2]
        for s, d in zip(ei[0], ei[1]):
            if st == "patent" and dt == "ipc" and int(s) in subset_pat:
                ipc_of_pat[int(s)].add(int(d))
            elif st == "ipc" and dt == "patent" and int(d) in subset_pat:
                ipc_of_pat[int(d)].add(int(s))

    # Collect all company/ipc nodes referenced by subset patents
    subset_co  = set()
    subset_ipc = set()
    for pi in subset_pat:
        subset_co.update(co_of_pat[pi])
        subset_ipc.update(ipc_of_pat[pi])

    # ── 4. Embedding similarity for all patents in subset ────────────────────
    vec_src = h_emb[src_idx]
    norm_src = np.linalg.norm(vec_src) + 1e-8
    emb_sim = {i: float(np.dot(h_emb[i], vec_src) / (np.linalg.norm(h_emb[i]) + 1e-8) / norm_src)
               for i in subset_pat}

    # ── 5. Build networkx for layout ─────────────────────────────────────────
    G = nx.DiGraph()
    for pi in subset_pat:
        G.add_node(f"pat_{pi}")
    for ci in subset_co:
        G.add_node(f"co_{ci}")
    for ii in subset_ipc:
        G.add_node(f"ipc_{ii}")

    # Add patent-patent edges (for layout physics)
    edge_records = []
    edge_id = 0
    for et in PAT_PAT:
        if et not in data.edge_index_dict:
            continue
        raw_type = et[1]
        relation = RELATION_MAP[raw_type]
        ei = data.edge_index_dict[et].cpu().numpy()
        for s, d in zip(ei[0], ei[1]):
            s, d = int(s), int(d)
            if s not in subset_pat or d not in subset_pat:
                continue
            sid, did = f"pat_{s}", f"pat_{d}"
            G.add_edge(sid, did)
            edge_records.append({
                "id": f"e{edge_id}", "source": sid, "target": did,
                "relation": relation, "raw_type": raw_type,
                "importance": 0.0, "attention": 0.0,
                "is_path": False, "is_counterfactual": False,
            })
            edge_id += 1

    # Add company edges
    for pi in subset_pat:
        for ci in co_of_pat.get(pi, []):
            sid, did = f"pat_{pi}", f"co_{ci}"
            G.add_edge(sid, did)
            edge_records.append({
                "id": f"e{edge_id}", "source": sid, "target": did,
                "relation": "owner", "raw_type": "owned_by",
                "importance": 0.5, "attention": 0.0,
                "is_path": False, "is_counterfactual": False,
            })
            edge_id += 1

    # Add IPC edges
    for pi in subset_pat:
        for ii in ipc_of_pat.get(pi, []):
            sid, did = f"pat_{pi}", f"ipc_{ii}"
            G.add_edge(sid, did)
            edge_records.append({
                "id": f"e{edge_id}", "source": sid, "target": did,
                "relation": "ipc", "raw_type": "has_ipc",
                "importance": 0.3, "attention": 0.0,
                "is_path": False, "is_counterfactual": False,
            })
            edge_id += 1

    # ── 6. Compute layout (fixed seed for reproducibility) ────────────────────
    # Give src/dst central positions as seeds
    pos_init = {}
    pos_init[f"pat_{src_idx}"] = np.array([0.0, 0.5])
    pos_init[f"pat_{dst_idx}"] = np.array([1.0, 0.5])

    pos = nx.spring_layout(G, k=1.5, seed=42, pos=pos_init, fixed=list(pos_init.keys()),
                           iterations=150)
    # Scale to pixel range
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    W, H = 800, 600

    def scale(p):
        x = float((p[0] - x_min) / (x_max - x_min + 1e-8) * W - W/2)
        y = float((p[1] - y_min) / (y_max - y_min + 1e-8) * H - H/2)
        return x, y

    # ── 7. Build node records ─────────────────────────────────────────────────
    node_records = []

    def add_node(nid, ntype, label, is_src=False, is_dst=False):
        p = pos.get(nid, np.array([0.0, 0.0]))
        x, y = scale(p)
        h_src = hop_src.get(int(nid.split("_")[1]), 99) if ntype == "patent" else 99
        h_dst = hop_dst.get(int(nid.split("_")[1]), 99) if ntype == "patent" else 99
        deg    = G.degree(nid) if nid in G else 0
        raw_i  = int(nid.split("_")[1])
        esim   = emb_sim.get(raw_i, 0.0) if ntype == "patent" else 0.0
        owner  = ""
        ipc_c  = []
        if ntype == "patent":
            co_idxs = list(co_of_pat.get(raw_i, []))
            owner   = co_list[co_idxs[0]] if co_idxs else ""
            ipc_idxs = list(ipc_of_pat.get(raw_i, []))
            ipc_c    = [ipc_list[ii] for ii in ipc_idxs[:3]]
        node_records.append({
            "id":         nid,
            "type":       ntype,
            "label":      label,
            "short_label": label[:14],
            "x": x, "y": y,
            "importance":         1.0 if (is_src or is_dst) else max(0.1, esim),
            "is_src":  is_src,
            "is_dst":  is_dst,
            "hop_from_src": h_src if h_src < 99 else None,
            "hop_from_dst": h_dst if h_dst < 99 else None,
            "min_hop": min(h_src, h_dst) if h_src < 99 or h_dst < 99 else None,
            "degree":   deg,
            "community": 0,
            "owner":    owner,
            "ipc":      ipc_c,
            "embedding_similarity": round(esim, 4),
        })

    for pi in subset_pat:
        add_node(f"pat_{pi}", "patent", pat_list[pi],
                 is_src=(pi == src_idx), is_dst=(pi == dst_idx))
    for ci in subset_co:
        p = pos.get(f"co_{ci}", np.array([0.0, 0.0]))
        x, y = scale(p)
        node_records.append({
            "id": f"co_{ci}", "type": "company",
            "label": co_list[ci], "short_label": co_list[ci][:14],
            "x": x, "y": y, "importance": 0.4,
            "is_src": False, "is_dst": False,
            "hop_from_src": None, "hop_from_dst": None, "min_hop": None,
            "degree": G.degree(f"co_{ci}") if f"co_{ci}" in G else 0,
            "community": 1, "owner": "", "ipc": [],
            "embedding_similarity": 0.0,
        })
    for ii in subset_ipc:
        p = pos.get(f"ipc_{ii}", np.array([0.0, 0.0]))
        x, y = scale(p)
        node_records.append({
            "id": f"ipc_{ii}", "type": "ipc",
            "label": ipc_list[ii], "short_label": ipc_list[ii][:8],
            "x": x, "y": y, "importance": 0.2,
            "is_src": False, "is_dst": False,
            "hop_from_src": None, "hop_from_dst": None, "min_hop": None,
            "degree": G.degree(f"ipc_{ii}") if f"ipc_{ii}" in G else 0,
            "community": 2, "owner": "", "ipc": [],
            "embedding_similarity": 0.0,
        })

    # ── 8. Top-N predicted defendants + predicted edges ──────────────────────
    PRED_N   = max(ctx.get("topk", 10), 20)
    logits   = h_emb @ h_emb[src_idx]
    logits[src_idx] = -np.inf
    probs    = 1 / (1 + np.exp(-logits))
    sort_idx = np.argsort(-probs)

    existing_ids = {nr["id"] for nr in node_records}
    pred_rank    = 0

    for idx in sort_idx:
        i = int(idx)
        if i == src_idx:
            continue
        if pred_rank >= PRED_N:
            break
        pred_rank += 1
        prob  = float(probs[i])
        nid   = f"pat_{i}"
        is_gt = (i == dst_idx)

        if nid not in existing_ids:
            angle = 2 * np.pi * (pred_rank - 1) / PRED_N
            ox    = float(W * 0.65 * np.cos(angle))
            oy    = float(H * 0.65 * np.sin(angle))
            co_idxs  = list(co_of_pat.get(i, []))
            ipc_idxs = list(ipc_of_pat.get(i, []))
            node_records.append({
                "id": nid, "type": "patent",
                "label": pat_list[i], "short_label": pat_list[i][:14],
                "x": ox, "y": oy,
                "importance": round(prob, 4),
                "is_src": False, "is_dst": is_gt,
                "is_predicted": True, "is_predicted_gt": is_gt,
                "hop_from_src": None, "hop_from_dst": None, "min_hop": None,
                "degree": 0, "community": 3,
                "owner": co_list[co_idxs[0]] if co_idxs else "",
                "ipc":   [ipc_list[ii] for ii in ipc_idxs[:3]],
                "embedding_similarity": round(prob, 4),
            })
            existing_ids.add(nid)
        else:
            for nr in node_records:
                if nr["id"] == nid:
                    nr["is_predicted"]    = True
                    nr["is_predicted_gt"] = is_gt
                    break

        edge_records.append({
            "id": f"pred_{pred_rank}", "source": f"pat_{src_idx}", "target": nid,
            "relation": "predicted", "raw_type": "predicted_sue",
            "importance": round(prob, 4), "attention": 0.0,
            "is_path": False, "is_counterfactual": False,
            "is_predicted": True, "is_ground_truth": is_gt,
            "predicted_rank": pred_rank, "probability": round(prob, 4),
        })

    # If ground truth is not in top-N, force-add it as a special predicted edge
    gt_in_pred = any(int(idx) == dst_idx for idx in sort_idx[:pred_rank + 1]
                     if int(idx) != src_idx)
    # Actually check edge_records
    gt_in_pred = any(e.get("is_ground_truth") for e in edge_records if e.get("is_predicted"))

    if not gt_in_pred:
        nid  = f"pat_{dst_idx}"
        prob = float(probs[dst_idx])
        gt_actual_rank = int((probs > prob).sum()) + 1

        if nid not in existing_ids:
            co_idxs  = list(co_of_pat.get(dst_idx, []))
            ipc_idxs = list(ipc_of_pat.get(dst_idx, []))
            node_records.append({
                "id": nid, "type": "patent",
                "label": pat_list[dst_idx], "short_label": pat_list[dst_idx][:14],
                "x": float(W * 0.55 * np.cos(np.pi * 0.25)),
                "y": float(H * 0.55 * np.sin(np.pi * 0.25)),
                "importance": round(prob, 4),
                "is_src": False, "is_dst": True,
                "is_predicted": True, "is_predicted_gt": True,
                "hop_from_src": None, "hop_from_dst": None, "min_hop": None,
                "degree": 0, "community": 3,
                "owner": co_list[co_idxs[0]] if co_idxs else "",
                "ipc":   [ipc_list[ii] for ii in ipc_idxs[:3]],
                "embedding_similarity": round(prob, 4),
            })
        else:
            for nr in node_records:
                if nr["id"] == nid:
                    nr["is_predicted"]    = True
                    nr["is_predicted_gt"] = True
                    break

        edge_records.append({
            "id": "pred_gt", "source": f"pat_{src_idx}", "target": nid,
            "relation": "predicted", "raw_type": "predicted_sue",
            "importance": round(prob, 4), "attention": 0.0,
            "is_path": False, "is_counterfactual": False,
            "is_predicted": True, "is_ground_truth": True,
            "predicted_rank": gt_actual_rank, "probability": round(prob, 4),
        })
        print(f"    GT forced at rank {gt_actual_rank}, P={prob:.4f}")

    print(f"    predicted_edges={pred_rank} (+ GT)")

    # ── 9. Save ───────────────────────────────────────────────────────────────
    result = {
        "metadata": {
            "pair": ctx["pair"], "khop": khop,
            "src_patent": ctx["src_label"], "dst_patent": ctx["dst_label"],
            "src_owners": ctx["src_owners"], "dst_owners": ctx["dst_owners"],
            "src_ipc":    ctx["src_ipc"],    "dst_ipc":    ctx["dst_ipc"],
            "p_base":     round(ctx["p_base"], 6),
            "node_count": len(node_records), "edge_count": len(edge_records),
            "predicted_count": pred_rank,
            "created_at": ctx["created_at"],
        },
        "nodes": node_records,
        "edges": edge_records,
    }
    save_json(result, "graph.json")
    print(f"    nodes={len(node_records)}, edges={len(edge_records)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--khop", type=int, default=2)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    ctx = load_context(args.pair, args.khop, args.topk)
    build_graph(ctx)
