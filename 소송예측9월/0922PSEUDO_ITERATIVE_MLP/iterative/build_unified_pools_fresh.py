"""
Build unified_pools/pool_{cutoff}.pkl from scratch.

Requires c3_data.pkl to already exist in fixed_data/c3_pools/cutoff_{cutoff}/.

Unlike build_unified_pools.py, this script does NOT depend on
'unified_pools_before_newpolicy/' — it builds source_meta directly from
c3_data.pkl.

Usage:
    python iterative/build_unified_pools_fresh.py --cutoff 2012 --target-start 2013 --target-end 2014
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from data import load_snapshot

TARGET_SIZE = 2000


def norm(x):
    if pd.isna(x):
        return None
    s = str(x)
    return s[:-2] if s.endswith(".0") else s


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def ipc_prefix(code, n):
    if code is None:
        return None
    s = str(code).strip().upper().replace(" ", "")
    return s[:n] if len(s) >= n else s


def build_graph_maps(obj):
    data = obj["data"]
    p2ipc4 = defaultdict(set)
    p2ipc3 = defaultdict(set)
    ipc4_to_pat = defaultdict(set)
    ipc3_to_pat = defaultdict(set)
    adj = defaultdict(set)
    ipc_list = [str(x) for x in obj.get("ipc_list", [])]

    et = ("patent", "has_ipc", "ipc")
    if et in data.edge_types:
        for p, i in data[et].edge_index.cpu().t().tolist():
            p, i = int(p), int(i)
            if i >= len(ipc_list):
                continue
            code = ipc_list[i]
            c4 = ipc_prefix(code, 4)
            c3 = ipc_prefix(code, 3)
            if c4:
                p2ipc4[p].add(c4)
                ipc4_to_pat[c4].add(p)
            if c3:
                p2ipc3[p].add(c3)
                ipc3_to_pat[c3].add(p)

    for et in [("patent", "cites", "patent"), ("patent", "cited_by", "patent")]:
        if et not in data.edge_types:
            continue
        for s, t in data[et].edge_index.cpu().t().tolist():
            s, t = int(s), int(t)
            adj[s].add(t)
            adj[t].add(s)

    return p2ipc4, p2ipc3, ipc4_to_pat, ipc3_to_pat, adj


def citation_layers(adj, src, max_hop=3):
    seen = {int(src)}
    frontier = {int(src)}
    exact = []
    for _ in range(max_hop):
        nxt = set()
        for x in frontier:
            nxt |= adj.get(x, set())
        nxt -= seen
        exact.append(nxt)
        seen |= nxt
        frontier = nxt
        if not frontier:
            exact += [set()] * (max_hop - len(exact))
            break
    c1 = exact[0] if len(exact) > 0 else set()
    c2 = c1 | (exact[1] if len(exact) > 1 else set())
    c3 = c2 | (exact[2] if len(exact) > 2 else set())
    return c1, c2, c3


def same_ipc(src, p2ipc, ipc_to_pat):
    out = set()
    for code in p2ipc.get(int(src), set()):
        out |= ipc_to_pat.get(code, set())
    out.discard(int(src))
    return out


def add_stage(selected, candidates, forbidden, limit):
    available = sorted(set(candidates) - forbidden - selected)
    room = max(0, limit - len(selected))
    added = available[:room]
    selected.update(added)
    return added


def build_fold(cutoff, ys, ye):
    print("\n" + "=" * 110)
    print(f"BUILD UNIFIED POOL (fresh) | {cutoff} -> {ys}-{ye}")
    print("=" * 110)

    snap = load_snapshot(config.SNAP_DIR, cutoff)
    p2i = {str(k): int(v) for k, v in snap["pat2idx"].items()}
    n_pat = len(p2i)

    p2ipc4, p2ipc3, ipc4_to_pat, ipc3_to_pat, adj = build_graph_maps(snap)

    # Load c3_data to get source patents and their real positives
    c3_path = ROOT / "fixed_data" / "c3_pools" / f"cutoff_{cutoff}" / "c3_data.pkl"
    c3 = load_pickle(c3_path)

    # Collect all srcs and real positives from c3_data
    real_positive_by_src = defaultdict(set)
    all_pos = c3.get("all_positive_pairs", set())
    for s, d in all_pos:
        real_positive_by_src[int(s)].add(int(d))

    # Build source_meta (minimal — no counterparty data needed here)
    source_meta = {}
    for rec in c3.get("train_records", []) + c3.get("val_records", []):
        src = int(rec["src"])
        if src not in source_meta:
            source_meta[src] = {"from_c3": True}

    all_src = sorted(source_meta)
    print(f"  total src patents: {len(all_src)}")

    pools = {}
    stage_pools = {}
    detail_rows = []

    for i, src in enumerate(all_src, 1):
        forbidden = {src} | real_positive_by_src.get(src, set())

        ipc4 = same_ipc(src, p2ipc4, ipc4_to_pat)
        ipc3 = same_ipc(src, p2ipc3, ipc3_to_pat)
        c1, c2, c3_cit = citation_layers(adj, src, 3)

        # No counterparty portfolio (no assignee info in source_meta)
        A1 = ipc3 & c1
        A2 = ipc3 & c2
        A3 = ipc3 & c3_cit

        selected = set()
        stages = {}

        stages["A1"] = add_stage(selected, A1, forbidden, TARGET_SIZE)
        stages["A2"] = add_stage(selected, A2, forbidden, TARGET_SIZE)
        stages["A3"] = add_stage(selected, A3, forbidden, TARGET_SIZE)

        n_after_hard = len(selected)

        stages["citation_1"] = add_stage(selected, c1, forbidden, TARGET_SIZE)
        stages["citation_2"] = add_stage(selected, c2, forbidden, TARGET_SIZE)
        stages["citation_3"] = add_stage(selected, c3_cit, forbidden, TARGET_SIZE)

        n_after_citation = len(selected)

        stages["ipc4"] = add_stage(selected, ipc4, forbidden, TARGET_SIZE)
        stages["ipc3"] = add_stage(selected, ipc3, forbidden, TARGET_SIZE)

        final_size = len(selected)

        pools[src] = sorted(selected)
        stage_pools[src] = {k: v for k, v in stages.items() if v}

        detail_rows.append(
            {
                "cutoff": cutoff,
                "target": f"{ys}-{ye}",
                "src": src,
                "n_ipc4": len(ipc4),
                "n_ipc3": len(ipc3),
                "n_citation1": len(c1),
                "n_citation2": len(c2),
                "n_citation3": len(c3_cit),
                "add_A1": len(stages["A1"]),
                "add_A2": len(stages["A2"]),
                "add_A3": len(stages["A3"]),
                "after_hard": n_after_hard,
                "add_citation1": len(stages["citation_1"]),
                "add_citation2": len(stages["citation_2"]),
                "add_citation3": len(stages["citation_3"]),
                "after_citation": n_after_citation,
                "add_ipc4": len(stages["ipc4"]),
                "add_ipc3": len(stages["ipc3"]),
                "final_pool": final_size,
                "still_short": final_size < TARGET_SIZE,
                "shortfall": max(0, TARGET_SIZE - final_size),
            }
        )

        if i % 200 == 0:
            print(f"{i:,}/{len(all_src):,} src processed")

    detail = pd.DataFrame(detail_rows)
    short = detail[detail["still_short"]].copy()

    out_dir = ROOT / "fixed_data" / "unified_pools"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "cutoff": cutoff,
        "target": [ys, ye],
        "pools": pools,
        "stage_pools": stage_pools,
        "real_positive_by_src": {s: sorted(v) for s, v in real_positive_by_src.items()},
        "source_meta": source_meta,
        "num_patents": n_pat,
        "policy": {
            "target_size": TARGET_SIZE,
            "ipc_hard_prefix": 3,
            "priority": ["A1", "A2", "A3", "citation_1", "citation_2", "citation_3", "ipc4", "ipc3"],
            "A1": "IPC3 ∩ citation1",
            "A2": "IPC3 ∩ citation<=2",
            "A3": "IPC3 ∩ citation<=3",
            "fallback": "citation1 -> citation<=2 -> citation<=3 -> IPC4 -> IPC3",
        },
    }

    save_pickle(payload, out_dir / f"pool_{cutoff}.pkl")
    detail.to_csv(out_dir / f"detail_{cutoff}.csv", index=False)
    short.to_csv(out_dir / f"still_short_{cutoff}.csv", index=False)

    print("\nRESULT")
    print(f"src              : {len(detail):,}")
    print(f"reached {TARGET_SIZE}     : {(detail.final_pool >= TARGET_SIZE).sum():,}")
    print(f"still short      : {len(short):,}")

    if len(short):
        print(f"  min : {short.final_pool.min():,.0f}")
        print(f"  mean: {short.final_pool.mean():,.1f}")

    print(f"\nsaved: {out_dir / f'pool_{cutoff}.pkl'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, required=True)
    ap.add_argument("--target-start", type=int, required=True)
    ap.add_argument("--target-end", type=int, required=True)
    args = ap.parse_args()
    build_fold(args.cutoff, args.target_start, args.target_end)
