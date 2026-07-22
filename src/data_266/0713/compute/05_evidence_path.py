"""compute/05_evidence_path.py → results/evidence_path.json
Find the best evidence path from src to dst using GNN importance + attention.
Also patches graph.json edges with is_path flag.
"""
import sys, json
from pathlib import Path
import numpy as np
import heapq

sys.path.insert(0, str(Path(__file__).parent))
from common import load_context, save_json, RESULTS_DIR


def build_evidence_path(ctx: dict) -> None:
    print("[05_evidence_path] Finding evidence path...")

    graph_path = RESULTS_DIR / "graph.json"
    if not graph_path.exists():
        raise FileNotFoundError("Run 00_graph.py first.")
    with open(graph_path) as f:
        graph = json.load(f)

    nodes     = graph["nodes"]
    edges     = graph["edges"]
    node_by_id = {n["id"]: n for n in nodes}

    src_id = f"pat_{ctx['src_idx']}"
    dst_id = f"pat_{ctx['dst_idx']}"

    # ── Combine importance + attention as path weight ─────────────────────────
    # Higher = better path edge
    def edge_weight(e):
        imp  = e.get("importance", 0.0)
        attn = e.get("attention", 0.0)
        return (imp + attn) / 2.0

    # Build adjacency: nid → [(score, nid_next, edge_id)]
    adj = {}
    for e in edges:
        s, d = e["source"], e["target"]
        w    = edge_weight(e)
        adj.setdefault(s, []).append((w, d, e["id"]))
        adj.setdefault(d, []).append((w, s, e["id"]))  # undirected search

    # ── Dijkstra (max-weight path) ────────────────────────────────────────────
    # visited set prevents cycles (양방향 그래프에서 무한루프 방지)
    dist    = {src_id: 0.0}
    prev    = {}
    visited = set()
    pq      = [(0.0, src_id)]   # (-weight, node)

    while pq:
        neg_d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        d = -neg_d
        for w, v, eid in adj.get(u, []):
            if v in visited:
                continue
            nd = d + w
            if nd > dist.get(v, -1.0):
                dist[v] = nd
                prev[v] = (u, eid)
                heapq.heappush(pq, (-nd, v))

    # ── Backtrack path ────────────────────────────────────────────────────────
    if dst_id not in dist:
        print("    No path found.")
        save_json({"path": [], "path_labels": [], "score": 0.0,
                   "hops": 0, "edges": []}, "evidence_path.json")
        return

    path_nodes = []
    path_edges = []
    cur = dst_id
    while cur != src_id:
        path_nodes.append(cur)
        if cur not in prev:
            break
        prev_nid, eid = prev[cur]
        e = next((e for e in edges if e["id"] == eid), None)
        if e:
            path_edges.append({
                "source":    e["source"],
                "target":    e["target"],
                "relation":  e["relation"],
                "importance": e.get("importance", 0.0),
                "attention":  e.get("attention", 0.0),
            })
        cur = prev_nid
    path_nodes.append(src_id)
    path_nodes.reverse()
    path_edges.reverse()

    path_labels = [node_by_id.get(n, {}).get("label", n) for n in path_nodes]
    score = round(dist.get(dst_id, 0.0), 4)

    # Patch graph.json
    path_edge_ids = {e["source"] + "_" + e["target"] for e in path_edges}
    for edge in edges:
        key = edge["source"] + "_" + edge["target"]
        edge["is_path"] = key in path_edge_ids

    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    print(f"    Path length={len(path_nodes)}  score={score}")

    result = {
        "path":        path_nodes,
        "path_labels": path_labels,
        "score":       score,
        "hops":        len(path_nodes) - 1,
        "edges":       path_edges,
        "created_at":  ctx["created_at"],
    }
    save_json(result, "evidence_path.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--khop", type=int, default=2)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    ctx = load_context(args.pair, args.khop, args.topk)
    build_evidence_path(ctx)
