from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from data import (
    EDGE_TYPES,
    StaticFeatureStore,
    load_snapshot,
    build_features,
)
from model import Model


CUTOFF = 2012

CHECKPOINT = (
    ROOT
    / "runs"
    / "cutoff_2010"
    / "round_02"
    / "checkpoint.pt"
)

INHERITED_PATH = (
    ROOT
    / "runs"
    / "final_test_2012"
    / "inherited_from_2010_R2.pkl"
)

TEST_GT_PATH = (
    ROOT
    / "fixed_data"
    / "test"
    / "test_gt_2013_2014.pkl"
)

OUT_DIR = (
    ROOT
    / "runs"
    / "final_test_2012"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHUNK_SIZE = 100000

HITS_K = [
    1,
    10,
    50,
    100,
    500,
    1000,
    10000,
]


# ============================================================
# Helpers
# ============================================================

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def make_edge_dict(snapshot):
    d = snapshot["data"]

    return {
        e: d[e].edge_index.cpu()
        for e in EDGE_TYPES
        if e in d.edge_types
    }


def make_model(feats):
    dims = {
        k: int(v.shape[1])
        for k, v in feats.items()
    }

    return Model(
        dims,
        config.HIDDEN,
        EDGE_TYPES,
        config.DROPOUT,
    )


def add_mp(ed, pairs):
    """
    2012 historical sue에
    inherited pseudo를 추가.

    중복 pair는 제거.
    """

    et = (
        "patent",
        "sue",
        "patent",
    )

    old = ed.get(
        et,
        torch.empty(
            (2, 0),
            dtype=torch.long,
        ),
    ).cpu()

    if old.numel():
        old = (
            torch.unique(
                old.t(),
                dim=0,
            )
            .t()
            .contiguous()
        )

    add_pairs = sorted(
        {
            (int(s), int(d))
            for s, d in pairs
        }
    )

    if add_pairs:
        add = torch.tensor(
            add_pairs,
            dtype=torch.long,
        ).t().contiguous()
    else:
        add = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

    merged = torch.cat(
        [old, add],
        dim=1,
    )

    if merged.numel():
        merged = (
            torch.unique(
                merged.t(),
                dim=0,
            )
            .t()
            .contiguous()
        )

    ed[et] = merged

    return {
        "historical":
            int(old.shape[1]),

        "pseudo_requested":
            len(add_pairs),

        "final":
            int(merged.shape[1]),

        "actually_new":
            int(merged.shape[1])
            - int(old.shape[1]),
    }


# ============================================================
# Full-ranking function
# ============================================================

@torch.no_grad()
def full_rank(
    model,
    z,
    src,
    true_dst,
    n_pat,
    blocked_dst,
    device,
):
    """
    Filtered Full Ranking

    후보:
      - 2012 snapshot의 모든 patent
      - src 자신 제외
      - 다른 known-positive dst 제외
      - true_dst는 반드시 유지

    rank:
      true score보다 높은 후보 수 + 1
    """

    src = int(src)
    true_dst = int(true_dst)

    # ----------------------------------------
    # True score
    # ----------------------------------------

    ts = torch.tensor(
        [src],
        dtype=torch.long,
        device=device,
    )

    td = torch.tensor(
        [true_dst],
        dtype=torch.long,
        device=device,
    )

    true_score = (
        model.score(
            z,
            ts,
            td,
        )
        .item()
    )

    # ----------------------------------------
    # Filter set
    # ----------------------------------------

    blocked = set(
        int(x)
        for x in blocked_dst
    )

    blocked.add(src)

    # 평가 대상 true dst는 반드시 살림
    blocked.discard(true_dst)

    higher = 0
    valid_candidates = 0

    # ----------------------------------------
    # 모든 patent를 chunk 단위 평가
    # ----------------------------------------

    for start in range(
        0,
        n_pat,
        CHUNK_SIZE,
    ):

        end = min(
            start + CHUNK_SIZE,
            n_pat,
        )

        cand = np.arange(
            start,
            end,
            dtype=np.int64,
        )

        if blocked:
            mask = np.array(
                [
                    int(x) not in blocked
                    for x in cand
                ],
                dtype=bool,
            )

            cand = cand[mask]

        if len(cand) == 0:
            continue

        dst = torch.as_tensor(
            cand,
            dtype=torch.long,
            device=device,
        )

        src_t = torch.full(
            (len(cand),),
            src,
            dtype=torch.long,
            device=device,
        )

        scores = model.score(
            z,
            src_t,
            dst,
        )

        higher += int(
            (scores > true_score)
            .sum()
            .item()
        )

        valid_candidates += len(cand)

    rank = 1 + higher

    return (
        rank,
        valid_candidates,
        true_score,
    )


# ============================================================
# Main
# ============================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device:",
        device,
    )

    # ========================================================
    # 1. 2012 snapshot
    # ========================================================

    print("\n[1] Load 2012 snapshot")

    snap = load_snapshot(
        config.SNAP_DIR,
        CUTOFF,
    )

    pat2idx = {
        str(k): int(v)
        for k, v
        in snap["pat2idx"].items()
    }

    idx2pat = {
        int(v): str(k)
        for k, v
        in pat2idx.items()
    }

    n_pat = len(pat2idx)

    print(
        "patent nodes:",
        f"{n_pat:,}",
    )

    # ========================================================
    # 2. Features
    # ========================================================

    print("\n[2] Build features")

    static = StaticFeatureStore.load(
        config.DATA_GNN
    )

    feats, info = build_features(
        snap,
        static,
    )

    ed = make_edge_dict(snap)

    # ========================================================
    # 3. 2010 R2 inherited pseudo
    # ========================================================

    print("\n[3] Add inherited pseudo MP")

    inherited = load_pkl(
        INHERITED_PATH
    )

    inherited_pairs = (
        inherited["inherited_pairs"]
    )

    mp_stats = add_mp(
        ed,
        inherited_pairs,
    )

    print(
        "historical GT MP :",
        f"{mp_stats['historical']:,}",
    )

    print(
        "inherited pseudo :",
        f"{mp_stats['pseudo_requested']:,}",
    )

    print(
        "actually new     :",
        f"{mp_stats['actually_new']:,}",
    )

    print(
        "final MP         :",
        f"{mp_stats['final']:,}",
    )

    # ========================================================
    # 4. GPU graph
    # ========================================================

    counts = {
        k: int(v.shape[0])
        for k, v in feats.items()
    }

    fd = {
        k: v.to(device)
        for k, v in feats.items()
    }

    gd = {
        "edge_dict": {
            k: v.to(device)
            for k, v in ed.items()
        },
        "counts": counts,
    }

    # ========================================================
    # 5. Model
    # ========================================================

    print("\n[4] Load 2010 R2 checkpoint")

    model = make_model(
        feats
    ).to(device)

    state = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        state
    )

    model.eval()

    print(
        "checkpoint:",
        CHECKPOINT,
    )

    # ========================================================
    # 6. Encode 2012 graph ONCE
    # ========================================================

    print("\n[5] Encode 2012 graph")

    with torch.no_grad():
        z = model.encode(
            fd,
            gd,
        )

    print(
        "patent embedding:",
        tuple(z.shape),
    )

    # ========================================================
    # 7. Test GT
    # ========================================================

    print("\n[6] Load Test GT")

    test_obj = load_pkl(
        TEST_GT_PATH
    )

    test_patent_pairs = sorted(
        {
            (str(a), str(b))
            for a, b
            in test_obj[
                "test_patent_pairs"
            ]
        }
    )

    print(
        "Test GT:",
        len(test_patent_pairs),
    )

    # patent ID -> 2012 index
    test_pairs = []

    missing_test = []

    for a, b in test_patent_pairs:

        if (
            a not in pat2idx
            or b not in pat2idx
        ):
            missing_test.append(
                (a, b)
            )
            continue

        test_pairs.append(
            (
                pat2idx[a],
                pat2idx[b],
                a,
                b,
            )
        )

    if missing_test:
        raise RuntimeError(
            "Test GT 중 2012 snapshot에 "
            f"없는 pair 발견: {missing_test[:10]}"
        )

    # ========================================================
    # 8. Known-positive filter
    # ========================================================
    #
    # Filter 대상:
    #
    # 2012 시점 graph의 모든 known sue positive
    # = historical GT MP + inherited pseudo
    #
    # + 다른 Test GT positive
    #
    # 같은 src에 여러 Test 정답이 존재하면,
    # 하나를 평가할 때 다른 정답 때문에
    # rank가 불필요하게 밀리는 것을 방지.
    # ========================================================

    print(
        "\n[7] Build filtered-positive map"
    )

    known_by_src = {}

    sue_et = (
        "patent",
        "sue",
        "patent",
    )

    final_sue = ed[sue_et]

    for s, d in (
        final_sue
        .t()
        .tolist()
    ):
        known_by_src.setdefault(
            int(s),
            set(),
        ).add(
            int(d)
        )

    # 다른 Test GT도 filtered positive
    for s, d, _, _ in test_pairs:
        known_by_src.setdefault(
            int(s),
            set(),
        ).add(
            int(d)
        )

    # ========================================================
    # 9. FULL RANKING
    # ========================================================

    print()
    print("=" * 80)
    print("FILTERED FULL RANKING")
    print("=" * 80)

    rows = []

    total = len(test_pairs)

    for i, (
        src,
        dst,
        plt_pat,
        def_pat,
    ) in enumerate(
        test_pairs,
        1,
    ):

        blocked = (
            known_by_src.get(
                src,
                set(),
            )
        )

        rank, n_candidates, score = (
            full_rank(
                model=model,
                z=z,
                src=src,
                true_dst=dst,
                n_pat=n_pat,
                blocked_dst=blocked,
                device=device,
            )
        )

        row = {
            "plt_patent":
                plt_pat,

            "def_patent":
                def_pat,

            "src":
                src,

            "dst":
                dst,

            "rank":
                rank,

            "n_candidates":
                n_candidates,

            "true_score":
                score,

            "RR":
                1.0 / rank,
        }

        for k in HITS_K:
            row[f"Hits@{k}"] = int(
                rank <= k
            )

        rows.append(row)

        print(
            f"[{i:03d}/{total}] "
            f"{plt_pat} -> {def_pat} | "
            f"rank={rank:,} / "
            f"{n_candidates:,}"
        )

    # ========================================================
    # 10. Metrics
    # ========================================================

    df = pd.DataFrame(rows)

    ranks = df["rank"].to_numpy(
        dtype=np.int64
    )

    metrics = {
        "cutoff":
            2012,

        "test_period":
            [2013, 2014],

        "source_model":
            "2010 R2",

        "ranking":
            "filtered_full",

        "test_pairs":
            int(len(df)),

        "patent_nodes":
            int(n_pat),

        "historical_gt_mp":
            int(mp_stats["historical"]),

        "inherited_pseudo_mp":
            int(
                mp_stats[
                    "pseudo_requested"
                ]
            ),

        "final_mp":
            int(mp_stats["final"]),

        "MRR":
            float(
                np.mean(
                    1.0 / ranks
                )
            ),

        "MeanRank":
            float(
                np.mean(ranks)
            ),

        "MedianRank":
            float(
                np.median(ranks)
            ),
    }

    for k in HITS_K:
        metrics[f"Hits@{k}"] = float(
            np.mean(
                ranks <= k
            )
        )

    # ========================================================
    # 11. Save
    # ========================================================

    csv_path = (
        OUT_DIR
        / "test_pair_ranks.csv"
    )

    json_path = (
        OUT_DIR
        / "final_test_metrics.json"
    )

    df.sort_values(
        "rank"
    ).to_csv(
        csv_path,
        index=False,
    )

    json_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # 12. Print
    # ========================================================

    print()
    print("=" * 80)
    print("FINAL TEST RESULT")
    print("=" * 80)

    print(
        "Test pairs  :",
        len(df),
    )

    print(
        "MRR         :",
        f"{metrics['MRR']:.8f}",
    )

    print(
        "Mean Rank   :",
        f"{metrics['MeanRank']:,.2f}",
    )

    print(
        "Median Rank :",
        f"{metrics['MedianRank']:,.2f}",
    )

    print()

    for k in HITS_K:
        print(
            f"Hits@{k:<5} : "
            f"{metrics[f'Hits@{k}']:.6f}"
        )

    print()
    print(
        "rank CSV :",
        csv_path,
    )

    print(
        "metrics  :",
        json_path,
    )


if __name__ == "__main__":
    main()
