from __future__ import annotations

import pickle
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch

import config


# ============================================================
# CONFIG
# ============================================================

FOLDS = [
    (2004, 2005, 2006),
    (2006, 2007, 2008),
    (2008, 2009, 2010),
    (2010, 2011, 2012),
]

TARGET = 1500

ROOT = config.EXP_DIR
SNAP_DIR = config.SNAP_DIR
OWNERSHIP_PATH = config.OWNERSHIP_PATH

# src 목록만 사용
UNIFIED_POOL_DIR = ROOT / "fixed_data" / "unified_pools"

OUT_DIR = ROOT / "results" / "ipc3_fallback_sequence_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# UTILS
# ============================================================

def norm(x):
    if pd.isna(x):
        return None

    s = str(x)

    if s.endswith(".0"):
        s = s[:-2]

    return s


def load_snapshot(year):
    path = SNAP_DIR / f"snapshot_{year}_v2.pt"

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


# ============================================================
# IPC LABEL
# ============================================================

def get_ipc_list(snap):

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
# IPC4 / IPC3 MAP
# ============================================================

def build_ipc_maps(snap):

    data = snap["data"]
    ipc_list = get_ipc_list(snap)

    p2ipc4 = defaultdict(set)
    ipc4_to_patents = defaultdict(set)

    p2ipc3 = defaultdict(set)
    ipc3_to_patents = defaultdict(set)

    et = ("patent", "has_ipc", "ipc")

    if et not in data.edge_types:
        return (
            p2ipc4,
            ipc4_to_patents,
            p2ipc3,
            ipc3_to_patents,
        )

    for p, ipc_idx in data[et].edge_index.cpu().t().tolist():

        p = int(p)
        ipc_idx = int(ipc_idx)

        ipc4 = str(ipc_list[ipc_idx]).strip()

        if not ipc4:
            continue

        ipc3 = ipc4[:3]

        p2ipc4[p].add(ipc4)
        ipc4_to_patents[ipc4].add(p)

        p2ipc3[p].add(ipc3)
        ipc3_to_patents[ipc3].add(p)

    return (
        p2ipc4,
        ipc4_to_patents,
        p2ipc3,
        ipc3_to_patents,
    )


def get_same_ipc(
    src,
    p2ipc,
    ipc_to_patents,
):

    result = set()

    for ipc in p2ipc.get(src, set()):
        result |= ipc_to_patents.get(ipc, set())

    result.discard(src)

    return result


# ============================================================
# CITATION
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

            adj[int(s)].add(int(t))

    return adj


def citation_layers(adj, src):

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

    oc = pd.read_parquet(
        OWNERSHIP_PATH
    ).copy()

    rename = {}

    if (
        "patent" in oc.columns
        and "patent_id" not in oc.columns
    ):
        rename["patent"] = "patent_id"

    if (
        "assignee" in oc.columns
        and "assignee_id" not in oc.columns
    ):
        rename["assignee"] = "assignee_id"

    oc = oc.rename(columns=rename)

    oc["patent_id"] = (
        oc["patent_id"].map(norm)
    )

    oc["assignee_id"] = (
        oc["assignee_id"].map(norm)
    )

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

    for patent in (
        q["patent_id"]
        .dropna()
        .astype(str)
        .unique()
    ):

        if patent in p2i:
            result.add(p2i[patent])

    return result


# ============================================================
# SRC -> COUNTERPARTY
# ============================================================

def build_src_counterparties(
    future,
    target_start,
    target_end,
):

    cm = future["case_meta"].copy()

    cm["date_filed"] = pd.to_datetime(
        cm["date_filed"],
        errors="coerce",
    )

    cm = cm[
        cm["date_filed"]
        .dt.year
        .between(
            target_start,
            target_end,
        )
    ]

    data = future["data"]

    pat_list = [
        str(x)
        for x in future["pat_list"]
    ]

    base_like_p2i = {
        str(k): int(v)
        for k, v in future["pat2idx"].items()
    }

    case_to_plt = defaultdict(set)

    et = (
        "case",
        "plaintiff_patent",
        "patent",
    )

    if et in data.edge_types:

        for c, p in (
            data[et]
            .edge_index
            .cpu()
            .t()
            .tolist()
        ):

            patent = pat_list[int(p)]

            case_to_plt[int(c)].add(patent)

    # 여기서는 patent number로 저장
    src_counterparties = defaultdict(set)

    for _, r in cm.iterrows():

        case_type = str(
            r.get("case_type", "")
        )

        if case_type not in {
            "case1",
            "case3",
        }:
            continue

        aid = norm(
            r.get("def_assignee_id")
        )

        if aid is None:
            continue

        node_idx = int(r["node_idx"])

        for patent in case_to_plt.get(
            node_idx,
            set(),
        ):

            src_counterparties[
                patent
            ].add(aid)

    return src_counterparties


# ============================================================
# UNIFIED SRC
# ============================================================

def load_unified_srcs(cutoff):

    path = (
        UNIFIED_POOL_DIR
        / f"pool_{cutoff}.pkl"
    )

    with open(path, "rb") as f:
        obj = pickle.load(f)

    return sorted(
        int(x)
        for x in obj["pools"].keys()
    )


# ============================================================
# ANALYZE ONE FOLD
# ============================================================

def analyze_fold(
    cutoff,
    ys,
    ye,
    ownership_groups,
):

    print()
    print("=" * 110)
    print(
        f"CUTOFF {cutoff} "
        f"-> {ys}-{ye}"
    )
    print("=" * 110)

    base = load_snapshot(cutoff)
    future = load_snapshot(ye)

    p2i = {
        str(k): int(v)
        for k, v in base["pat2idx"].items()
    }

    idx2pat = {
        int(v): str(k)
        for k, v in p2i.items()
    }

    srcs = load_unified_srcs(cutoff)

    (
        p2ipc4,
        ipc4_to_patents,
        p2ipc3,
        ipc3_to_patents,
    ) = build_ipc_maps(base)

    adj = build_citation_adj(base)

    src_counterparties = (
        build_src_counterparties(
            future,
            ys,
            ye,
        )
    )

    cutoff_dt = pd.Timestamp(
        f"{cutoff}-12-30"
    )

    rows = []

    for i, src in enumerate(srcs, 1):

        # ====================================================
        # IPC
        # ====================================================

        same_ipc4 = get_same_ipc(
            src,
            p2ipc4,
            ipc4_to_patents,
        )

        same_ipc3 = get_same_ipc(
            src,
            p2ipc3,
            ipc3_to_patents,
        )

        # ====================================================
        # CITATION
        # ====================================================

        h1, h2, h3 = citation_layers(
            adj,
            src,
        )

        upto1 = set(h1)
        upto2 = h1 | h2
        upto3 = h1 | h2 | h3

        # ====================================================
        # COUNTERPARTY PORTFOLIO
        # ====================================================

        patent_number = idx2pat.get(src)

        counterparties = (
            src_counterparties.get(
                patent_number,
                set(),
            )
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
        # ORIGINAL IPC3 HARD
        # ====================================================

        A1 = same_ipc3 & upto1
        A2 = same_ipc3 & upto2
        A3 = same_ipc3 & upto3

        B1 = portfolio & (
            same_ipc3 | upto1
        )

        B2 = portfolio & (
            same_ipc3 | upto2
        )

        B3 = portfolio & (
            same_ipc3 | upto3
        )

        original = (
            A1
            | A2
            | A3
            | B1
            | B2
            | B3
        )

        original.discard(src)

        # ====================================================
        # FALLBACK SEQUENCE
        #
        # 실제로 앞 단계에서 없었던 후보만 추가됨.
        # set union이므로 중복 자동 제거.
        # ====================================================

        stage0 = set(original)

        stage1 = stage0 | upto1
        stage1.discard(src)

        stage2 = stage1 | upto2
        stage2.discard(src)

        stage3 = stage2 | upto3
        stage3.discard(src)

        stage4 = stage3 | same_ipc4
        stage4.discard(src)

        stage5 = stage4 | same_ipc3
        stage5.discard(src)

        # ====================================================
        # 어느 단계에서 처음 1500 도달?
        # ====================================================

        sizes = [
            len(stage0),
            len(stage1),
            len(stage2),
            len(stage3),
            len(stage4),
            len(stage5),
        ]

        names = [
            "original",
            "citation_1",
            "citation_2",
            "citation_3",
            "ipc4",
            "ipc3",
        ]

        reached_stage = "still_short"

        for name, size in zip(
            names,
            sizes,
        ):

            if size >= TARGET:
                reached_stage = name
                break

        rows.append({
            "cutoff": cutoff,
            "src": src,

            "original":
                len(stage0),

            "after_citation1":
                len(stage1),

            "after_citation2":
                len(stage2),

            "after_citation3":
                len(stage3),

            "after_ipc4":
                len(stage4),

            "after_ipc3":
                len(stage5),

            "gain_citation1":
                len(stage1) - len(stage0),

            "gain_citation2":
                len(stage2) - len(stage1),

            "gain_citation3":
                len(stage3) - len(stage2),

            "gain_ipc4":
                len(stage4) - len(stage3),

            "gain_ipc3":
                len(stage5) - len(stage4),

            "reached_stage":
                reached_stage,

            "final_ge_1500":
                len(stage5) >= TARGET,
        })

        if i % 250 == 0:
            print(
                f"processed "
                f"{i:,}/{len(srcs):,}"
            )

    df = pd.DataFrame(rows)

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("STAGE COUNTS")
    print("-" * 80)

    print(
        df["reached_stage"]
        .value_counts()
        .reindex(
            [
                "original",
                "citation_1",
                "citation_2",
                "citation_3",
                "ipc4",
                "ipc3",
                "still_short",
            ],
            fill_value=0,
        )
    )

    print()
    print("CUMULATIVE >=1500")
    print("-" * 80)

    columns = [
        ("original", "original"),
        ("+ citation1", "after_citation1"),
        ("+ citation2", "after_citation2"),
        ("+ citation3", "after_citation3"),
        ("+ IPC4", "after_ipc4"),
        ("+ IPC3", "after_ipc3"),
    ]

    for label, col in columns:

        count = int(
            (df[col] >= TARGET).sum()
        )

        pct = (
            count / len(df) * 100
        )

        print(
            f"{label:15s}: "
            f"{count:5,d} / {len(df):5,d} "
            f"({pct:6.2f}%)"
        )

    print()
    print("FINAL AFTER IPC3")
    print("-" * 80)

    x = df["after_ipc3"]

    print(f"mean   : {x.mean():,.1f}")
    print(f"median : {x.median():,.1f}")
    print(f"min    : {x.min():,.0f}")
    print(f"P10    : {x.quantile(.10):,.1f}")
    print(f"P25    : {x.quantile(.25):,.1f}")
    print(f"P75    : {x.quantile(.75):,.1f}")
    print(f"P90    : {x.quantile(.90):,.1f}")
    print(f"max    : {x.max():,.0f}")

    print()
    print(
        "still <1500:",
        int((x < TARGET).sum()),
    )

    # ========================================================
    # SAVE
    # ========================================================

    path = (
        OUT_DIR
        / f"fallback_sequence_{cutoff}.csv"
    )

    df.to_csv(
        path,
        index=False,
    )

    print("saved:", path)

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    ownership_groups = load_ownership()

    dfs = []

    for cutoff, ys, ye in FOLDS:

        df = analyze_fold(
            cutoff,
            ys,
            ye,
            ownership_groups,
        )

        dfs.append(df)

    all_df = pd.concat(
        dfs,
        ignore_index=True,
    )

    # ========================================================
    # OVERALL / BY FOLD SUMMARY
    # ========================================================

    summary_rows = []

    for cutoff, x in all_df.groupby(
        "cutoff",
        sort=True,
    ):

        n = len(x)

        summary_rows.append({
            "cutoff": cutoff,

            "src_count": n,

            "original_ge1500":
                int(
                    (x["original"] >= TARGET)
                    .sum()
                ),

            "after_c1_ge1500":
                int(
                    (
                        x["after_citation1"]
                        >= TARGET
                    ).sum()
                ),

            "after_c2_ge1500":
                int(
                    (
                        x["after_citation2"]
                        >= TARGET
                    ).sum()
                ),

            "after_c3_ge1500":
                int(
                    (
                        x["after_citation3"]
                        >= TARGET
                    ).sum()
                ),

            "after_ipc4_ge1500":
                int(
                    (
                        x["after_ipc4"]
                        >= TARGET
                    ).sum()
                ),

            "after_ipc3_ge1500":
                int(
                    (
                        x["after_ipc3"]
                        >= TARGET
                    ).sum()
                ),

            "still_short":
                int(
                    (
                        x["after_ipc3"]
                        < TARGET
                    ).sum()
                ),

            "median_final":
                float(
                    x["after_ipc3"]
                    .median()
                ),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    print()
    print("=" * 110)
    print("ALL FOLDS")
    print("=" * 110)

    print(
        summary.to_string(
            index=False
        )
    )

    all_path = (
        OUT_DIR
        / "fallback_sequence_all.csv"
    )

    summary_path = (
        OUT_DIR
        / "fallback_sequence_summary.csv"
    )

    all_df.to_csv(
        all_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("all    :", all_path)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()