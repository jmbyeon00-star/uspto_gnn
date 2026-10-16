from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
C3_DIR = ROOT / "fixed_data" / "c3_pools"


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def patent_pair(r):
    return str(r["plt_patent"]), str(r["def_patent"])


def pair_set(df: pd.DataFrame):
    if len(df) == 0:
        return set()
    return set(
        zip(
            df["src"].astype(int),
            df["dst"].astype(int),
        )
    )


def run(
    cutoff: int,
    target_start: int,
    target_end: int,
    round_no: int,
    threshold: float,
):
    next_round = round_no + 1

    round_dir = (
        RUNS_DIR
        / f"cutoff_{cutoff}"
        / f"round_{round_no:02d}"
    )

    top1_path = round_dir / "top1_candidates.parquet"
    round_data_path = round_dir / "round_data.pkl"
    c3_path = (
        C3_DIR
        / f"cutoff_{cutoff}"
        / "c3_data.pkl"
    )

    if not top1_path.exists():
        raise FileNotFoundError(top1_path)

    if not round_data_path.exists():
        raise FileNotFoundError(round_data_path)

    if not c3_path.exists():
        raise FileNotFoundError(c3_path)

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    df = pd.read_parquet(top1_path)
    round_data = load_pickle(round_data_path)
    c3 = load_pickle(c3_path)

    required = {
        "case_row_id",
        "case_type",
        "date_filed",
        "event_year",
        "pseudo_bucket",
        "known_patent",
        "top1_patent",
        "plt_patent",
        "def_patent",
        "src",
        "dst",
        "top1_score",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"top1_candidates missing columns: {sorted(missing)}"
        )

    # ------------------------------------------------------------------
    # BASIC CONSISTENCY
    # ------------------------------------------------------------------

    if int(round_data["cutoff"]) != cutoff:
        raise RuntimeError(
            f"round_data cutoff mismatch: "
            f"{round_data['cutoff']} != {cutoff}"
        )

    if list(round_data["target"]) != [target_start, target_end]:
        raise RuntimeError(
            f"round_data target mismatch: "
            f"{round_data['target']} != "
            f"{[target_start, target_end]}"
        )

    # ------------------------------------------------------------------
    # 1. THRESHOLD
    # ------------------------------------------------------------------

    selected = df[
        df["top1_score"] >= threshold
    ].copy()

    n_input = len(df)
    n_threshold = len(selected)

    # ------------------------------------------------------------------
    # 2. BASIC VALIDITY
    # ------------------------------------------------------------------

    selected = selected[
        selected["src"].notna()
        & selected["dst"].notna()
        & selected["event_year"].notna()
    ].copy()

    selected["src"] = selected["src"].astype(int)
    selected["dst"] = selected["dst"].astype(int)
    selected["event_year"] = selected["event_year"].astype(int)

    invalid_self_mask = selected["src"] == selected["dst"]
    n_self_removed = int(invalid_self_mask.sum())

    selected = selected[~invalid_self_mask].copy()

    # ------------------------------------------------------------------
    # 3. TEMPORAL WINDOW
    #
    # MP  : event <= cutoff
    # SUP : target_start <= event <= target_end
    # else: not usable in this fold
    # ------------------------------------------------------------------

    temporal_bucket = pd.Series(
        "outside",
        index=selected.index,
        dtype="object",
    )

    temporal_bucket.loc[
        selected["event_year"] <= cutoff
    ] = "mp"

    temporal_bucket.loc[
        selected["event_year"].between(
            target_start,
            target_end,
            inclusive="both",
        )
    ] = "sup"

    selected["selected_bucket"] = temporal_bucket

    # Verify score_case12's existing pseudo_bucket agrees.
    expected_existing = selected["pseudo_bucket"].astype(str)

    bucket_mismatch = (
        (
            selected["selected_bucket"].isin(["mp", "sup"])
        )
        & (
            expected_existing
            != selected["selected_bucket"]
        )
    )

    n_bucket_mismatch = int(bucket_mismatch.sum())

    if n_bucket_mismatch:
        bad = selected.loc[
            bucket_mismatch,
            [
                "case_row_id",
                "event_year",
                "pseudo_bucket",
                "selected_bucket",
            ],
        ]

        raise RuntimeError(
            "pseudo_bucket temporal mismatch detected:\n"
            + bad.head(20).to_string(index=False)
        )

    outside = selected[
        selected["selected_bucket"] == "outside"
    ].copy()

    selected = selected[
        selected["selected_bucket"].isin(["mp", "sup"])
    ].copy()

    n_outside = len(outside)

    # ------------------------------------------------------------------
    # 4. REMOVE DUPLICATE PAIRS
    #
    # If multiple Case1/2 cases infer the same patent pair,
    # retain the highest-scoring one.
    # ------------------------------------------------------------------

    before_dup = len(selected)

    selected = (
        selected
        .sort_values(
            ["top1_score", "date_filed", "case_row_id"],
            ascending=[False, True, True],
        )
        .drop_duplicates(
            subset=["plt_patent", "def_patent"],
            keep="first",
        )
        .copy()
    )

    n_duplicate_removed = before_dup - len(selected)

    # ------------------------------------------------------------------
    # 5. FIXED C3 PROTECTION
    # ------------------------------------------------------------------

    real_positive_pairs = {
        (int(s), int(d))
        for s, d in c3["all_positive_pairs"]
    }

    val_reserved_pairs = {
        (int(s), int(d))
        for s, d in c3["val_reserved_pairs"]
    }

    selected["_pair"] = list(zip(selected["src"], selected["dst"]))
    selected["_patent_pair"] = list(zip(
        selected["plt_patent"].astype(str),
        selected["def_patent"].astype(str),
    ))

    real_collision_mask = selected["_pair"].isin(
        real_positive_pairs
    )

    n_real_collision = int(real_collision_mask.sum())

    selected = selected[
        ~real_collision_mask
    ].copy()

    val_collision_mask = selected["_pair"].isin(
        val_reserved_pairs
    )

    n_val_collision = int(val_collision_mask.sum())

    selected = selected[
        ~val_collision_mask
    ].copy()

    # ------------------------------------------------------------------
    # 6. PREVIOUS / INHERITED PSEUDO PROTECTION
    # Cross-round/fold identity comparison is ALWAYS patent-ID based.
    # ------------------------------------------------------------------

    previous_inherited = round_data.get("mp_inherited", [])
    previous_mp = round_data.get("mp_pseudo", [])
    previous_sup = round_data.get("sup_pseudo", [])

    previous_patent_pairs = set()
    for r in previous_inherited + previous_mp + previous_sup:
        if isinstance(r, dict) and "plt_patent" in r and "def_patent" in r:
            previous_patent_pairs.add(patent_pair(r))

    previous_collision_mask = selected["_patent_pair"].isin(previous_patent_pairs)
    n_previous_collision = int(previous_collision_mask.sum())
    selected = selected[~previous_collision_mask].copy()

    # ------------------------------------------------------------------
    # 7. FINAL MP / SUP
    # ------------------------------------------------------------------

    mp_df = selected[
        selected["selected_bucket"] == "mp"
    ].copy()

    sup_df = selected[
        selected["selected_bucket"] == "sup"
    ].copy()

    # Keep useful metadata, but records remain directly usable through
    # record["src"] / record["dst"].
    keep_cols = [
        "case_row_id",
        "case_type",
        "date_filed",
        "event_year",
        "selected_bucket",
        "missing_side",
        "known_patent",
        "top1_patent",
        "plt_patent",
        "def_patent",
        "src",
        "dst",
        "top1_score",
        "top2_score",
        "top2_patent",
        "margin",
    ]

    keep_cols = [
        c for c in keep_cols
        if c in selected.columns
    ]

    def to_records(x: pd.DataFrame):
        y = x[keep_cols].copy()

        if "date_filed" in y.columns:
            y["date_filed"] = y["date_filed"].astype(str)

        return y.to_dict("records")

    mp_records = to_records(mp_df)
    sup_records = to_records(sup_df)

    # ------------------------------------------------------------------
    # 8. FINAL VERIFICATION
    # ------------------------------------------------------------------

    mp_pairs = pair_set(mp_df)
    sup_pairs = pair_set(sup_df)
    all_selected_pairs = mp_pairs | sup_pairs

    checks = {
        "mp_sup_overlap":
            len(mp_pairs & sup_pairs),

        "selected_real_positive_overlap":
            len(all_selected_pairs & real_positive_pairs),

        "selected_val_reserved_overlap":
            len(all_selected_pairs & val_reserved_pairs),

        "selected_previous_pseudo_overlap":
            len(set(selected['_patent_pair']) & previous_patent_pairs),

        "mp_future_violation":
            int((mp_df["event_year"] > cutoff).sum()),

        "sup_year_violation":
            int(
                (
                    ~sup_df["event_year"].between(
                        target_start,
                        target_end,
                        inclusive="both",
                    )
                ).sum()
            ),

        "duplicate_final_pairs":
            (
                len(mp_df)
                + len(sup_df)
                - len(all_selected_pairs)
            ),
    }

    if any(v != 0 for v in checks.values()):
        print("\nFAILED CHECKS")
        for k, v in checks.items():
            print(f"{k:<40}: {v:,}")

        raise RuntimeError(
            "Pseudo selection verification failed."
        )

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    stats = {
        "top1_queries": n_input,
        "threshold_pass": n_threshold,
        "self_pair_removed": n_self_removed,
        "outside_temporal_window": n_outside,
        "duplicate_pair_removed": n_duplicate_removed,
        "real_positive_collision_removed": n_real_collision,
        "val_reserved_collision_removed": n_val_collision,
        "previous_pseudo_collision_removed":
            n_previous_collision,
        "pseudo_mp_selected": len(mp_records),
        "pseudo_sup_selected": len(sup_records),
        "total_selected":
            len(mp_records) + len(sup_records),
    }

    output = {
        "cutoff": cutoff,
        "target": [target_start, target_end],
        "source_round": round_no,
        "next_round": next_round,
        "threshold": threshold,

        "pseudo_mp": mp_records,
        "pseudo_sup": sup_records,

        "stats": stats,
        "verification": checks,

        "sources": {
            "top1_candidates": str(top1_path),
            "round_data": str(round_data_path),
            "c3_data": str(c3_path),
        },
    }

    out_pkl = (
        round_dir
        / f"pseudo_selected_for_round_{next_round:02d}.pkl"
    )

    out_csv = (
        round_dir
        / f"pseudo_selected_for_round_{next_round:02d}.csv"
    )

    save_pickle(output, out_pkl)

    selected_out = pd.concat(
        [mp_df, sup_df],
        ignore_index=True,
    )

    selected_out.drop(
        columns=["_pair", "_patent_pair"],
        errors="ignore",
    ).to_csv(
        out_csv,
        index=False,
    )

    # ------------------------------------------------------------------
    # PRINT
    # ------------------------------------------------------------------

    print("=" * 100)
    print(
        f"SELECT PSEUDO | cutoff={cutoff} | "
        f"round {round_no} -> {next_round} | "
        f"threshold={threshold}"
    )
    print("=" * 100)

    print("\nINPUT")
    print(f"Top1 queries                 : {n_input:,}")
    print(f"score >= {threshold:<8}       : {n_threshold:,}")

    print("\nFILTER")
    print(f"self pair removed            : {n_self_removed:,}")
    print(f"outside temporal window      : {n_outside:,}")
    print(f"duplicate pair removed       : {n_duplicate_removed:,}")
    print(f"real positive collision      : {n_real_collision:,}")
    print(f"VAL reserved collision       : {n_val_collision:,}")
    print(
        f"previous pseudo collision    : "
        f"{n_previous_collision:,}"
    )

    print("\nSELECTED")
    print(f"pseudo MP                    : {len(mp_records):,}")
    print(f"pseudo SUP                   : {len(sup_records):,}")
    print(
        f"TOTAL                        : "
        f"{len(mp_records) + len(sup_records):,}"
    )

    print("\nVERIFICATION")

    for k, v in checks.items():
        print(f"{k:<40}: {v:,}")

    print("\nVERIFICATION PASSED")

    print("\nsaved:")
    print(out_pkl)
    print(out_csv)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--cutoff",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--target-start",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--target-end",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--round",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--threshold",
        type=float,
        default=0.99,
    )

    args = ap.parse_args()

    run(
        cutoff=args.cutoff,
        target_start=args.target_start,
        target_end=args.target_end,
        round_no=args.round,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()