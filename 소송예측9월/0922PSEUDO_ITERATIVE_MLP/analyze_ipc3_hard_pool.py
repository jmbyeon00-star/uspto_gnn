from __future__ import annotations

import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

import config


# ============================================================
# CONFIG
# ============================================================

CUTOFFS = [
    (2004, 2005, 2006),
    (2006, 2007, 2008),
    (2008, 2009, 2010),
    (2010, 2011, 2012),
]

ROOT = config.EXP_DIR
SNAP_DIR = config.SNAP_DIR
OWNERSHIP_PATH = config.OWNERSHIP_PATH

# 현재 unified pool에서 "어떤 src를 대상으로 하는지"만 가져옴.
# pool 내용 자체는 사용하지 않음.
UNIFIED_POOL_DIR = ROOT / "fixed_data" / "unified_pools"

OUT_DIR = ROOT / "results" / "ipc3_original_hard_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# BASIC UTILS
# ============================================================

def norm(x):
    if pd.isna(x):
        return None

    s = str(x)

    if s.endswith(".0"):
        s = s[:-2]

    return s


def load_snapshot(cutoff):
    path = SNAP_DIR / f"snapshot_{cutoff}_v2.pt"

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


# ============================================================
# IPC LABEL
# ============================================================

def get_ipc_list(snap):
    """
    snapshot의 IPC node index -> IPC4 label.
    예:
        1019 -> D01B
        xxxx -> G06F
    """

    if "ipc_list" in snap:
        return [str(x) for x in snap["ipc_list"]]

    if "ipc_codes" in snap:
        return [str(x) for x in snap["ipc_codes"]]

    if "idx2ipc" in snap:
        x = snap["idx2ipc"]

        if isinstance(x, dict):
            max_idx = max(int(k) for k in x.keys())
            arr = [None] * (max_idx + 1)

            for k, v in x.items():
                arr[int(k)] = str(v)

            return arr

    raise RuntimeError(
        "snapshot에서 IPC label mapping을 찾지 못했습니다."
    )


# ============================================================
# IPC3 MAP
#
# 기존:
#   G06F
#   G06Q
#
# 변경:
#   G06
#   G06
#
# patent가 IPC 여러 개면 합집합 방식.
# ============================================================

def build_ipc3_maps(snap):
    data = snap["data"]
    ipc_list = get_ipc_list(snap)

    p2ipc3 = defaultdict(set)
    ipc3_to_patents = defaultdict(set)

    et = ("patent", "has_ipc", "ipc")

    if et not in data.edge_types:
        return p2ipc3, ipc3_to_patents

    for p, ipc_idx in data[et].edge_index.cpu().t().tolist():

        p = int(p)
        ipc_idx = int(ipc_idx)

        ipc4 = str(ipc_list[ipc_idx]).strip()

        if not ipc4:
            continue

        ipc3 = ipc4[:3]

        p2ipc3[p].add(ipc3)
        ipc3_to_patents[ipc3].add(p)

    return p2ipc3, ipc3_to_patents


# ============================================================
# CITATION GRAPH
# ============================================================

def build_citation_adj(snap):
    data = snap["data"]

    adj = defaultdict(set)

    for et in [
        ("patent", "cites", "patent"),
        ("patent", "cited_by", "patent"),
    ]:

        if et not in data.edge_types:
            continue

        for s, t in data[et].edge_index.cpu().t().tolist():

            s = int(s)
            t = int(t)

            adj[s].add(t)

    return adj


def citation_layers(adj, src):
    """
    exact 1-hop / exact 2-hop / exact 3-hop
    """

    src = int(src)

    seen = {src}
    frontier = {src}

    layers = []

    for _ in range(3):

        nxt = set()

        for node in frontier:
            nxt |= adj.get(node, set())

        nxt -= seen

        layers.append(nxt)

        seen |= nxt
        frontier = nxt

        if not frontier:

            while len(layers) < 3:
                layers.append(set())

            break

    return layers[0], layers[1], layers[2]


# ============================================================
# OWNERSHIP
# ============================================================

def load_ownership():
    oc = pd.read_parquet(OWNERSHIP_PATH).copy()

    rename = {}

    if "patent" in oc.columns and "patent_id" not in oc.columns:
        rename["patent"] = "patent_id"

    if "assignee" in oc.columns and "assignee_id" not in oc.columns:
        rename["assignee"] = "assignee_id"

    oc = oc.rename(columns=rename)

    oc["patent_id"] = oc["patent_id"].map(norm)
    oc["assignee_id"] = oc["assignee_id"].map(norm)

    oc["start_dt"] = pd.to_datetime(
        oc["start_dt"],
        errors="coerce",
    )

    oc["end_dt"] = pd.to_datetime(
        oc["end_dt"],
        errors="coerce",
    )

    return {
        str(a): g
        for a, g in oc.groupby(
            "assignee_id",
            sort=False,
        )
    }


def get_portfolio(
    ownership_groups,
    assignee_id,
    cutoff_dt,
    p2i,
):
    aid = norm(assignee_id)

    if aid is None:
        return set()

    if aid not in ownership_groups:
        return set()

    q = ownership_groups[aid]

    q = q[
        (
            q["start_dt"].isna()
            | q["start_dt"].le(cutoff_dt)
        )
        &
        (
            q["end_dt"].isna()
            | q["end_dt"].gt(cutoff_dt)
        )
    ]

    result = set()

    for patent in q["patent_id"].dropna().astype(str).unique():

        if patent in p2i:
            result.add(p2i[patent])

    return result


# ============================================================
# CASE -> COUNTERPARTY ASSIGNEE
#
# 같은 src가 여러 case에서 사용될 수 있으므로
# src -> counterparty assignee set 으로 저장
# ============================================================

def build_src_counterparties(
    snap,
    target_start,
    target_end,
):
    """
    future snapshot의 case_meta를 이용하여
    src별 상대방 assignee를 찾는다.

    Case1:
        known plaintiff patent = src
        counterparty = defendant company

    Case3:
        plaintiff patent = src
        counterparty = defendant company

    Case2는 방향상 known defendant patent이므로
    여기 src 목록 생성에는 사용하지 않는다.
    """

    case_meta = snap["case_meta"].copy()

    case_meta["date_filed"] = pd.to_datetime(
        case_meta["date_filed"],
        errors="coerce",
    )

    case_meta = case_meta[
        case_meta["date_filed"]
        .dt.year
        .between(target_start, target_end)
    ]

    data = snap["data"]

    pat_list = [
        str(x)
        for x in snap["pat_list"]
    ]

    pat2idx = {
        str(k): int(v)
        for k, v in snap["pat2idx"].items()
    }

    case_to_plaintiff_patents = defaultdict(set)

    et = (
        "case",
        "plaintiff_patent",
        "patent",
    )

    if et in data.edge_types:

        for c, p in data[et].edge_index.cpu().t().tolist():

            patent = pat_list[int(p)]

            if patent in pat2idx:
                case_to_plaintiff_patents[int(c)].add(
                    pat2idx[patent]
                )

    src_counterparties = defaultdict(set)

    for _, r in case_meta.iterrows():

        case_type = str(r.get("case_type", ""))

        if case_type not in {"case1", "case3"}:
            continue

        node_idx = int(r["node_idx"])

        defendant_assignee = norm(
            r.get("def_assignee_id")
        )

        if defendant_assignee is None:
            continue

        for src in case_to_plaintiff_patents.get(
            node_idx,
            set(),
        ):
            src_counterparties[src].add(
                defendant_assignee
            )

    return src_counterparties


# ============================================================
# LOAD UNIFIED SRC
# ============================================================

def load_unified_srcs(cutoff):
    """
    현재 unified pool에서 src key만 가져온다.
    기존 pool 내용은 절대 변경하지 않는다.
    """

    path = (
        UNIFIED_POOL_DIR
        / f"pool_{cutoff}.pkl"
    )

    with open(path, "rb") as f:
        obj = pickle.load(f)

    pools = obj["pools"]

    return sorted(
        int(x)
        for x in pools.keys()
    )


# ============================================================
# ONE FOLD
# ============================================================

def analyze_fold(
    cutoff,
    target_start,
    target_end,
    ownership_groups,
):

    print()
    print("=" * 110)
    print(
        f"CUTOFF {cutoff} "
        f"-> {target_start}-{target_end}"
    )
    print("=" * 110)

    # --------------------------------------------------------
    # cutoff snapshot
    # --------------------------------------------------------

    base = load_snapshot(cutoff)

    # target_end snapshot은 case 정보 확인용
    future = load_snapshot(target_end)

    p2i = {
        str(k): int(v)
        for k, v in base["pat2idx"].items()
    }

    # --------------------------------------------------------
    # 모든 unified src
    # --------------------------------------------------------

    srcs = load_unified_srcs(cutoff)

    print("unified src:", len(srcs))

    # --------------------------------------------------------
    # IPC3
    # --------------------------------------------------------

    p2ipc3, ipc3_to_patents = build_ipc3_maps(base)

    # --------------------------------------------------------
    # citation
    # --------------------------------------------------------

    adj = build_citation_adj(base)

    # --------------------------------------------------------
    # counterparty
    # --------------------------------------------------------

    src_counterparties = build_src_counterparties(
        future,
        target_start,
        target_end,
    )

    cutoff_dt = pd.Timestamp(
        f"{cutoff}-12-30"
    )

    rows = []

    # --------------------------------------------------------
    # SRC LOOP
    # --------------------------------------------------------

    for i, src in enumerate(srcs, 1):

        # ====================================================
        # IPC3 candidates
        # ====================================================

        src_ipc3 = p2ipc3.get(src, set())

        same_ipc3 = set()

        for ipc3 in src_ipc3:

            same_ipc3 |= ipc3_to_patents.get(
                ipc3,
                set(),
            )

        same_ipc3.discard(src)

        # ====================================================
        # citation 1/2/3
        # ====================================================

        h1, h2, h3 = citation_layers(
            adj,
            src,
        )

        upto1 = h1
        upto2 = h1 | h2
        upto3 = h1 | h2 | h3

        # ====================================================
        # A1 / A2 / A3
        # ====================================================

        A1 = same_ipc3 & upto1
        A2 = same_ipc3 & upto2
        A3 = same_ipc3 & upto3

        # ====================================================
        # counterparty portfolio
        #
        # 같은 src가 여러 case에 있으면 상대 회사 portfolio
        # 전체 합집합
        # ====================================================

        counterparties = src_counterparties.get(
            src,
            set(),
        )

        portfolio = set()

        for aid in counterparties:

            portfolio |= get_portfolio(
                ownership_groups,
                aid,
                cutoff_dt,
                p2i,
            )

        # ====================================================
        # B1 / B2 / B3
        # ====================================================

        B1 = portfolio & (
            same_ipc3 | upto1
        )

        B2 = portfolio & (
            same_ipc3 | upto2
        )

        B3 = portfolio & (
            same_ipc3 | upto3
        )

        # self 제거
        for x in [
            A1, A2, A3,
            B1, B2, B3,
        ]:
            x.discard(src)

        # ====================================================
        # UNIQUE HARD
        #
        # 아직 positive/reserved 제거 X
        # 순수 IPC3 정의의 크기만 보는 진단
        # ====================================================

        hard_union = (
            A1
            | A2
            | A3
            | B1
            | B2
            | B3
        )

        rows.append({
            "cutoff": cutoff,
            "target_start": target_start,
            "target_end": target_end,

            "src": src,

            "n_ipc3_codes":
                len(src_ipc3),

            "ipc3_codes":
                "|".join(sorted(src_ipc3)),

            "same_ipc3":
                len(same_ipc3),

            "citation_1":
                len(upto1),

            "citation_upto2":
                len(upto2),

            "citation_upto3":
                len(upto3),

            "A1_ipc3_x_c1":
                len(A1),

            "A2_ipc3_x_c2":
                len(A2),

            "A3_ipc3_x_c3":
                len(A3),

            "n_counterparties":
                len(counterparties),

            "counterparty_portfolio":
                len(portfolio),

            "B1_portfolio_x_ipc3_or_c1":
                len(B1),

            "B2_portfolio_x_ipc3_or_c2":
                len(B2),

            "B3_portfolio_x_ipc3_or_c3":
                len(B3),

            "hard_union":
                len(hard_union),
        })

        if i % 250 == 0:
            print(
                f"processed "
                f"{i:,}/{len(srcs):,}"
            )

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # SAVE SRC DETAIL
    # --------------------------------------------------------

    detail_path = (
        OUT_DIR
        / f"ipc3_hard_detail_{cutoff}.csv"
    )

    df.to_csv(
        detail_path,
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    hard = df["hard_union"]

    summary = {
        "cutoff": cutoff,
        "target":
            f"{target_start}-{target_end}",

        "src_count":
            len(df),

        "mean":
            float(hard.mean()),

        "median":
            float(hard.median()),

        "min":
            int(hard.min()),

        "p10":
            float(hard.quantile(.10)),

        "p25":
            float(hard.quantile(.25)),

        "p75":
            float(hard.quantile(.75)),

        "p90":
            float(hard.quantile(.90)),

        "max":
            int(hard.max()),

        "empty":
            int((hard == 0).sum()),

        "lt_100":
            int((hard < 100).sum()),

        "lt_500":
            int((hard < 500).sum()),

        "lt_1000":
            int((hard < 1000).sum()),

        "lt_1500":
            int((hard < 1500).sum()),

        "ge_1500":
            int((hard >= 1500).sum()),
    }

    print()
    print("IPC3 HARD POOL")
    print("-" * 70)

    print(f"src count : {summary['src_count']:,}")
    print(f"mean      : {summary['mean']:,.1f}")
    print(f"median    : {summary['median']:,.1f}")
    print(f"min       : {summary['min']:,}")
    print(f"P10       : {summary['p10']:,.1f}")
    print(f"P25       : {summary['p25']:,.1f}")
    print(f"P75       : {summary['p75']:,.1f}")
    print(f"P90       : {summary['p90']:,.1f}")
    print(f"max       : {summary['max']:,}")

    print()

    n = len(df)

    for label, key in [
        ("empty", "empty"),
        ("< 100", "lt_100"),
        ("< 500", "lt_500"),
        ("< 1,000", "lt_1000"),
        ("< 1,500", "lt_1500"),
        (">= 1,500", "ge_1500"),
    ]:

        value = summary[key]

        print(
            f"{label:10s}: "
            f"{value:,} "
            f"({value / n * 100:.2f}%)"
        )

    print()
    print("saved:", detail_path)

    return df, summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 110)
    print("IPC3 ORIGINAL HARD-POOL ANALYSIS")
    print("=" * 110)

    print()
    print("A1 = IPC3 ∩ Citation 1-hop")
    print("A2 = IPC3 ∩ Citation <=2-hop")
    print("A3 = IPC3 ∩ Citation <=3-hop")

    print()

    print(
        "B1 = Counterparty portfolio "
        "∩ (IPC3 ∪ Citation 1-hop)"
    )

    print(
        "B2 = Counterparty portfolio "
        "∩ (IPC3 ∪ Citation <=2-hop)"
    )

    print(
        "B3 = Counterparty portfolio "
        "∩ (IPC3 ∪ Citation <=3-hop)"
    )

    print()

    print(
        "No fallback / no pool overwrite."
    )

    ownership_groups = load_ownership()

    all_details = []
    summaries = []

    for cutoff, ys, ye in CUTOFFS:

        df, summary = analyze_fold(
            cutoff,
            ys,
            ye,
            ownership_groups,
        )

        all_details.append(df)
        summaries.append(summary)

    # --------------------------------------------------------
    # 전체 detail
    # --------------------------------------------------------

    all_df = pd.concat(
        all_details,
        ignore_index=True,
    )

    all_path = (
        OUT_DIR
        / "ipc3_hard_detail_all_folds.csv"
    )

    all_df.to_csv(
        all_path,
        index=False,
    )

    # --------------------------------------------------------
    # fold summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summaries
    )

    summary_path = (
        OUT_DIR
        / "ipc3_hard_summary_by_fold.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 110)
    print("ALL FOLDS SUMMARY")
    print("=" * 110)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("detail :", all_path)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()

    