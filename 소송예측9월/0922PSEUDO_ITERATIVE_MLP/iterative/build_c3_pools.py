"""
Build c3_data.pkl for a single (cutoff, ys, ye) fold.

Output: fixed_data/c3_pools/cutoff_{cutoff}/c3_data.pkl
        fixed_data/c3_pools/cutoff_{cutoff}/pool_stats.csv
        fixed_data/c3_pools/cutoff_{cutoff}/summary.json
        fixed_data/val_sets/val_{cutoff}.pkl

Usage:
    python iterative/build_c3_pools.py --cutoff 2012 --target-start 2013 --target-end 2014
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from data import load_snapshot


def norm(x):
    if pd.isna(x):
        return None
    s = str(x)
    return s[:-2] if s.endswith(".0") else s


def relation_maps(obj):
    d = obj["data"]
    p2ipc, ipc2p, adj = {}, {}, {}

    if ("patent", "has_ipc", "ipc") in d.edge_types:
        for p, i in d["patent", "has_ipc", "ipc"].edge_index.cpu().t().tolist():
            p2ipc.setdefault(int(p), set()).add(int(i))
            ipc2p.setdefault(int(i), set()).add(int(p))

    for et in [("patent", "cites", "patent"), ("patent", "cited_by", "patent")]:
        if et in d.edge_types:
            for s, t in d[et].edge_index.cpu().t().tolist():
                adj.setdefault(int(s), set()).add(int(t))

    return p2ipc, ipc2p, adj


def khop(adj, src, k):
    seen = {int(src)}
    frontier = {int(src)}
    layers = []

    for _ in range(k):
        nxt = set()
        for x in frontier:
            nxt |= adj.get(x, set())
        nxt -= seen
        layers.append(nxt)
        seen |= nxt
        frontier = nxt
        if not frontier:
            layers += [set()] * (k - len(layers))
            break

    return layers


def ownership_groups():
    oc = pd.read_parquet(config.OWNERSHIP_PATH).copy()
    ren = {}
    if "patent" in oc.columns and "patent_id" not in oc.columns:
        ren["patent"] = "patent_id"
    if "assignee" in oc.columns and "assignee_id" not in oc.columns:
        ren["assignee"] = "assignee_id"
    oc = oc.rename(columns=ren)
    oc["patent_id"] = oc["patent_id"].map(norm)
    oc["assignee_id"] = oc["assignee_id"].map(norm)
    oc["start_dt"] = pd.to_datetime(oc["start_dt"], errors="coerce")
    oc["end_dt"] = pd.to_datetime(oc["end_dt"], errors="coerce")
    return {str(a): g for a, g in oc.groupby("assignee_id", sort=False)}


def portfolio(groups, aid, cutoff_dt, p2i):
    aid = norm(aid)
    if aid is None or aid not in groups:
        return set()
    q = groups[aid]
    q = q[
        (q.start_dt.isna() | q.start_dt.le(cutoff_dt))
        & (q.end_dt.isna() | q.end_dt.gt(cutoff_dt))
    ]
    return {p2i[str(p)] for p in q.patent_id.dropna().astype(str).unique() if str(p) in p2i}


def future_case3(base, future, ys, ye):
    cm = future["case_meta"].copy()
    cm["date_filed"] = pd.to_datetime(cm.date_filed, errors="coerce")
    cm = cm[
        (cm.case_type.astype(str) == "case3")
        & cm.date_filed.dt.year.between(ys, ye)
    ]
    fpat = [str(x) for x in future["pat_list"]]
    p2i = {str(k): int(v) for k, v in base["pat2idx"].items()}
    cplt, cdef = {}, {}
    d = future["data"]
    for et, target in [
        (("case", "plaintiff_patent", "patent"), cplt),
        (("case", "defendant_patent", "patent"), cdef),
    ]:
        if et in d.edge_types:
            for c, p in d[et].edge_index.cpu().t().tolist():
                target.setdefault(int(c), set()).add(fpat[int(p)])

    rows = []
    for _, r in cm.iterrows():
        ni = int(r.node_idx)
        pl = sorted(x for x in cplt.get(ni, set()) if x in p2i)
        de = sorted(x for x in cdef.get(ni, set()) if x in p2i)
        if not pl or not de:
            continue
        rows.append(
            dict(
                case_row_id=int(r.case_row_id),
                date_filed=r.date_filed,
                def_assignee_id=r.get("def_assignee_id"),
                plt=pl,
                de=de,
            )
        )
    return rows


def split_half(items, rng):
    a = np.array(sorted(set(items)), dtype=np.int64)
    if len(a) == 0:
        return set(), set()
    rng.shuffle(a)
    n = len(a) // 2
    return set(a[:n].tolist()), set(a[n:].tolist())


def main(cutoff, ys, ye):
    rng = np.random.default_rng(config.SEED + cutoff)

    print(f"Loading base snapshot ({cutoff}) ...")
    base = load_snapshot(config.SNAP_DIR, cutoff)
    print(f"Loading future snapshot ({ye}) ...")
    future = load_snapshot(config.SNAP_DIR, ye)

    p2i = {str(k): int(v) for k, v in base["pat2idx"].items()}
    n_pat = len(p2i)

    p2ipc, ipc2p, adj = relation_maps(base)
    og = ownership_groups()
    cutoff_dt = pd.Timestamp(f"{cutoff}-12-30")

    print("Extracting case3 pairs ...")
    cases = future_case3(base, future, ys, ye)
    print(f"  eligible case3 cases: {len(cases)}")

    ids = np.array([x["case_row_id"] for x in cases], dtype=np.int64)
    rng.shuffle(ids)
    ntr = int(len(ids) * config.TRAIN_RATIO)
    tr_ids = set(ids[:ntr].tolist())
    va_ids = set(ids[ntr:].tolist())

    all_pos = set()
    for c in cases:
        for s in c["plt"]:
            for d in c["de"]:
                all_pos.add((p2i[s], p2i[d]))

    # include historical positives from base snapshot
    if ("patent", "sue", "patent") in base["data"].edge_types:
        all_pos |= set(
            map(tuple, base["data"]["patent", "sue", "patent"].edge_index.cpu().t().tolist())
        )

    train_records = []
    val_records = []
    val_reserved = set()
    stats = []

    for c in cases:
        case_pos = {p2i[x] for x in c["de"]}
        pf = portfolio(og, c["def_assignee_id"], cutoff_dt, p2i)

        for sp in c["plt"]:
            s = p2i[sp]
            same = set()
            for ip in p2ipc.get(s, set()):
                same |= ipc2p.get(ip, set())

            layers = khop(adj, s, 3)
            c1 = layers[0] if len(layers) > 0 else set()
            c2 = c1 | (layers[1] if len(layers) > 1 else set())
            c3 = c2 | (layers[2] if len(layers) > 2 else set())

            forbid = {s} | case_pos | {d for ss, d in all_pos if ss == s}
            A = (same & c1) - forbid
            B = (same & c2) - forbid - A
            C = (same & c3) - forbid - A - B
            D = (pf & (same | c3)) - forbid - A - B - C
            tiers = [A, B, C, D]
            halves = [split_half(x, rng) for x in tiers]
            train_half = set().union(*(x[0] for x in halves))
            val_half = set().union(*(x[1] for x in halves))

            for dp in c["de"]:
                d = p2i[dp]
                rec = {"case_row_id": c["case_row_id"], "src": s, "dst": d}

                if c["case_row_id"] in va_ids:
                    vh = list(val_half)
                    rng.shuffle(vh)
                    hard = vh[: config.N_HARD_VAL]
                    blocked = {s, d} | {x for ss, x in all_pos if ss == s} | set(hard)
                    randoms = []
                    while len(hard) + len(randoms) < config.N_TOTAL_VAL:
                        x = int(rng.integers(0, n_pat))
                        if x not in blocked and (s, x) not in val_reserved:
                            randoms.append(x)
                            blocked.add(x)
                    chosen = set(hard) | set(randoms)
                    val_reserved |= {(s, x) for x in chosen}
                    final_train_hard = train_half | (val_half - set(hard))
                    rec.update(hard=hard, random=randoms, final_train_hard=sorted(final_train_hard))
                    val_records.append(rec)
                else:
                    rec.update(final_train_hard=sorted(train_half | val_half))
                    train_records.append(rec)

            stats.append(
                {
                    "case_row_id": c["case_row_id"],
                    "src": s,
                    "n_A": len(A),
                    "n_B": len(B),
                    "n_C": len(C),
                    "n_D": len(D),
                    "n_hard": len(A | B | C | D),
                }
            )

    out = ROOT / "fixed_data" / "c3_pools" / f"cutoff_{cutoff}"
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "cutoff": cutoff,
        "target": [ys, ye],
        "train_case_ids": sorted(tr_ids),
        "val_case_ids": sorted(va_ids),
        "train_records": train_records,
        "val_records": val_records,
        "val_reserved_pairs": val_reserved,
        "all_positive_pairs": all_pos,
        "num_patents": n_pat,
    }

    with open(out / "c3_data.pkl", "wb") as f:
        pickle.dump(payload, f)

    pd.DataFrame(stats).to_csv(out / "pool_stats.csv", index=False)

    summary = {
        "cutoff": cutoff,
        "target": f"{ys}-{ye}",
        "eligible_cases": len(cases),
        "train_cases": len(tr_ids),
        "val_cases": len(va_ids),
        "train_pairs": len(train_records),
        "val_pairs": len(val_records),
        "val_reserved_pairs": len(val_reserved),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    val_dir = ROOT / "fixed_data" / "val_sets"
    val_dir.mkdir(parents=True, exist_ok=True)
    with open(val_dir / f"val_{cutoff}.pkl", "wb") as f:
        pickle.dump(val_records, f)

    print(json.dumps(summary, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, required=True)
    ap.add_argument("--target-start", type=int, required=True)
    ap.add_argument("--target-end", type=int, required=True)
    a = ap.parse_args()
    main(a.cutoff, a.target_start, a.target_end)
