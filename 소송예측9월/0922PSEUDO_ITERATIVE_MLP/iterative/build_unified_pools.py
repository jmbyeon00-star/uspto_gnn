from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
sys.path.insert(0, str(EXP_DIR))

import config
from data import load_snapshot

TARGET_SIZE = 2000


# ============================================================
# UTILS
# ============================================================

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


# ============================================================
# GRAPH MAPS
# ============================================================

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

    # citation은 양방향 neighborhood로 취급
    for et in [
        ("patent", "cites", "patent"),
        ("patent", "cited_by", "patent"),
    ]:
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


# ============================================================
# OWNERSHIP
# ============================================================

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

    return {
        p2i[str(p)]
        for p in q.patent_id.dropna().astype(str).unique()
        if str(p) in p2i
    }


# ============================================================
# CASE1 / CASE3 SRC + COUNTERPARTY
# ============================================================

def find_case12_base(cutoff):
    candidates = [
        EXP_DIR / "fixed_data" / "case12_base" / f"case12_base_{cutoff}.pkl",
        EXP_DIR / "fixed_data" / "case12_base" / f"pool_{cutoff}.pkl",
        EXP_DIR / "fixed_data" / "case12_base" / f"cutoff_{cutoff}.pkl",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def extract_src_meta_from_old_pool(cutoff):
    """
    이전 unified pool의 source_meta를 재사용.
    여기에는 C3 / Case1 src 및 counterparty assignee 정보가
    이미 들어 있다고 가정한다.
    """

    old_dir = EXP_DIR / "fixed_data" / "unified_pools_before_newpolicy"
    candidates = [
        old_dir / f"pool_{cutoff}.pkl",
        old_dir / f"unified_pool_{cutoff}.pkl",
        old_dir / f"cutoff_{cutoff}.pkl",
    ]

    for path in candidates:
        if not path.exists():
            continue

        obj = load_pickle(path)
        if "source_meta" not in obj:
            continue

        return obj["source_meta"], obj

    raise FileNotFoundError(
        f"cutoff={cutoff}: 기존 pool의 source_meta를 찾지 못했습니다."
    )


# ============================================================
# POSITIVES
# ============================================================

def real_positive_by_src_from_old(old_obj):
    raw = old_obj.get("real_positive_by_src", {})
    return {
        int(src): {int(x) for x in dsts}
        for src, dsts in raw.items()
    }


# ============================================================
# SOURCE META PARSING
# ============================================================

def get_counterparty_ids(meta):
    """
    source_meta 내부 key 이름이 약간 달라도 대응.
    한 src가 여러 case에 등장하면 counterparty 여러 개 가능.
    """

    keys = [
        "counterparty_assignee_ids",
        "counterparty_ids",
        "def_assignee_ids",
        "counterparties",
    ]

    for key in keys:
        if key not in meta:
            continue

        x = meta[key]

        if x is None:
            return []

        if isinstance(x, (list, tuple, set)):
            return [norm(v) for v in x if norm(v) is not None]

        return [norm(x)]

    # 단일 def assignee
    for key in ["def_assignee_id", "counterparty_assignee_id"]:
        if key in meta and norm(meta[key]) is not None:
            return [norm(meta[key])]

    return []


# ============================================================
# PRIORITY ADD
# ============================================================

def add_stage(selected, candidates, forbidden, limit):
    """
    candidates에서
      - forbidden
      - 이미 selected
    를 제외하고 최대 limit까지 추가.
    """

    available = sorted(set(candidates) - forbidden - selected)

    room = max(0, limit - len(selected))
    added = available[:room]

    selected.update(added)
    return added


# ============================================================
# BUILD ONE FOLD
# ============================================================

def build_fold(cutoff, ys, ye):
    print("\n" + "=" * 110)
    print(f"BUILD NEW UNIFIED POOL | {cutoff} -> {ys}-{ye}")
    print("=" * 110)

    snap = load_snapshot(config.SNAP_DIR, cutoff)
    p2i = {str(k): int(v) for k, v in snap["pat2idx"].items()}
    n_pat = len(p2i)

    p2ipc4, p2ipc3, ipc4_to_pat, ipc3_to_pat, adj = build_graph_maps(snap)
    ownership = ownership_groups()
    cutoff_dt = pd.Timestamp(f"{cutoff}-12-30")

    source_meta, old_obj = extract_src_meta_from_old_pool(cutoff)
    real_pos_by_src = real_positive_by_src_from_old(old_obj)

    # source_meta key를 int로 통일
    source_meta = {int(k): v for k, v in source_meta.items()}
    all_src = sorted(source_meta)

    pools = {}
    stage_pools = {}
    detail_rows = []

    for i, src in enumerate(all_src, 1):
        meta = source_meta[src]
        forbidden = {src} | real_pos_by_src.get(src, set())

        ipc4 = same_ipc(src, p2ipc4, ipc4_to_pat)
        ipc3 = same_ipc(src, p2ipc3, ipc3_to_pat)
        c1, c2, c3 = citation_layers(adj, src, 3)

        # 해당 src와 연결된 모든 counterparty portfolio의 합집합
        cp_portfolio = set()
        for aid in get_counterparty_ids(meta):
            cp_portfolio |= portfolio(ownership, aid, cutoff_dt, p2i)

        # ----------------------------------------------------
        # HARD: B 먼저
        # ----------------------------------------------------
        B1 = cp_portfolio & (ipc3 | c1)
        B2 = cp_portfolio & (ipc3 | c2)
        B3 = cp_portfolio & (ipc3 | c3)

        # ----------------------------------------------------
        # HARD: A 다음
        # ----------------------------------------------------
        A1 = ipc3 & c1
        A2 = ipc3 & c2
        A3 = ipc3 & c3

        selected = set()
        stages = {}

        stages["B1"] = add_stage(selected, B1, forbidden, TARGET_SIZE)
        stages["B2"] = add_stage(selected, B2, forbidden, TARGET_SIZE)
        stages["B3"] = add_stage(selected, B3, forbidden, TARGET_SIZE)

        stages["A1"] = add_stage(selected, A1, forbidden, TARGET_SIZE)
        stages["A2"] = add_stage(selected, A2, forbidden, TARGET_SIZE)
        stages["A3"] = add_stage(selected, A3, forbidden, TARGET_SIZE)

        n_after_hard = len(selected)

        # ----------------------------------------------------
        # FALLBACK
        # citation 먼저, IPC 나중
        # ----------------------------------------------------
        stages["citation_1"] = add_stage(selected, c1, forbidden, TARGET_SIZE)
        stages["citation_2"] = add_stage(selected, c2, forbidden, TARGET_SIZE)
        stages["citation_3"] = add_stage(selected, c3, forbidden, TARGET_SIZE)

        n_after_citation = len(selected)

        stages["ipc4"] = add_stage(selected, ipc4, forbidden, TARGET_SIZE)
        stages["ipc3"] = add_stage(selected, ipc3, forbidden, TARGET_SIZE)

        final_size = len(selected)

        pools[src] = sorted(selected)
        stage_pools[src] = {k: v for k, v in stages.items() if v}

        detail_rows.append({
            "cutoff": cutoff,
            "target": f"{ys}-{ye}",
            "src": src,
            "n_counterparties": len(get_counterparty_ids(meta)),
            "n_portfolio": len(cp_portfolio),
            "n_ipc4": len(ipc4),
            "n_ipc3": len(ipc3),
            "n_citation1": len(c1),
            "n_citation2": len(c2),
            "n_citation3": len(c3),

            "add_B1": len(stages["B1"]),
            "add_B2": len(stages["B2"]),
            "add_B3": len(stages["B3"]),
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
        })

        if i % 500 == 0:
            print(f"{i:,}/{len(all_src):,} src processed")

    detail = pd.DataFrame(detail_rows)
    short = detail[detail["still_short"]].copy()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------
    out_dir = EXP_DIR / "fixed_data" / "unified_pools"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "cutoff": cutoff,
        "target": [ys, ye],
        "pools": pools,
        "stage_pools": stage_pools,
        "real_positive_by_src": real_pos_by_src,
        "source_meta": source_meta,
        "num_patents": n_pat,
        "policy": {
            "target_size": TARGET_SIZE,
            "ipc_hard_prefix": 3,
            "priority": [
                "B1", "B2", "B3",
                "A1", "A2", "A3",
                "citation_1", "citation_2", "citation_3",
                "ipc4", "ipc3",
            ],
            "B1": "counterparty portfolio ∩ (IPC3 ∪ citation1)",
            "B2": "counterparty portfolio ∩ (IPC3 ∪ citation<=2)",
            "B3": "counterparty portfolio ∩ (IPC3 ∪ citation<=3)",
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
    print(f"reached 1500     : {(detail.final_pool >= TARGET_SIZE).sum():,}")
    print(f"still short      : {len(short):,}")

    if len(short):
        print("\nSTILL SHORT POOL SIZE")
        print(f"min              : {short.final_pool.min():,.0f}")
        print(f"mean             : {short.final_pool.mean():,.1f}")
        print(f"median           : {short.final_pool.median():,.1f}")
        print(f"max              : {short.final_pool.max():,.0f}")

    print(f"\nsaved: {out_dir / f'pool_{cutoff}.pkl'}")
    print(f"detail: {out_dir / f'detail_{cutoff}.csv'}")
    print(f"short : {out_dir / f'still_short_{cutoff}.csv'}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, required=True)
    ap.add_argument("--target-start", type=int, required=True)
    ap.add_argument("--target-end", type=int, required=True)
    args = ap.parse_args()

    build_fold(args.cutoff, args.target_start, args.target_end)