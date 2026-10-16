from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"

THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
PERCENTILES = [0.90, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999]


def pct(n, total):
    return 100.0 * n / total if total else 0.0


def threshold_table(df):
    rows = []
    total = len(df)

    for t in THRESHOLDS:
        hit = df["score"] >= t
        x = df.loc[hit]

        mp = int((x["pseudo_bucket"] == "mp").sum())
        sup = int((x["pseudo_bucket"] == "sup").sum())
        c1 = int((x["case_type"] == "case1").sum())
        c2 = int((x["case_type"] == "case2").sum())

        rows.append({
            "threshold": t,
            "ALL": len(x),
            "ALL_pct": pct(len(x), total),
            "MP": mp,
            "SUP": sup,
            "CASE1": c1,
            "CASE2": c2,
        })

    return pd.DataFrame(rows)


def percentile_table(df):
    s = df["score"].dropna()
    vals = s.quantile(PERCENTILES)

    return pd.DataFrame({
        "percentile": [
            "p90", "p95", "p97.5", "p99",
            "p99.5", "p99.9", "p99.95", "p99.99"
        ],
        "score": vals.values,
    })


def group_summary(df, col):
    rows = []
    for name, g in df.groupby(col):
        s = g["score"].dropna()
        rows.append({
            "group": name,
            "count": len(s),
            "mean": s.mean(),
            "median": s.median(),
            "p95": s.quantile(.95),
            "p99": s.quantile(.99),
            "p99.5": s.quantile(.995),
            "p99.9": s.quantile(.999),
            "max": s.max(),
        })
    return pd.DataFrame(rows)


def rank1_summary(df):
    if "rank" not in df.columns:
        return None

    x = df[df["rank"] == 1].copy()
    if x.empty:
        return None

    s = x["score"]
    return {
        "count": len(x),
        "mean": s.mean(),
        "median": s.median(),
        "p90": s.quantile(.90),
        "p95": s.quantile(.95),
        "p99": s.quantile(.99),
        "max": s.max(),
    }


def main(cutoff, round_idx):
    rdir = RUNS_DIR / f"cutoff_{cutoff}" / f"round_{round_idx:02d}"
    path = rdir / "scored_candidates.parquet"

    if not path.exists():
        raise FileNotFoundError(f"scored_candidates 없음: {path}")

    df = pd.read_parquet(path)

    required = {"score", "pseudo_bucket", "case_type"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"필수 컬럼 없음: {sorted(missing)}")

    print("=" * 110)
    print(f"ANALYZE SCORES | cutoff={cutoff} | round={round_idx}")
    print("=" * 110)
    print(f"file           : {path}")
    print(f"candidate rows : {len(df):,}")
    print(f"score min/max  : {df['score'].min():.10g} / {df['score'].max():.10g}")

    th = threshold_table(df)

    print("\n" + "=" * 110)
    print("ABSOLUTE SIGMOID THRESHOLDS")
    print("=" * 110)
    print(
        th.to_string(
            index=False,
            formatters={
                "threshold": lambda x: f">= {x:.2f}",
                "ALL_pct": lambda x: f"{x:.6f}%",
            },
        )
    )

    pt = percentile_table(df)

    print("\n" + "=" * 110)
    print("PERCENTILES — ALL CANDIDATE SCORES")
    print("=" * 110)
    print(
        pt.to_string(
            index=False,
            formatters={"score": lambda x: f"{x:.10g}"}
        )
    )

    print("\n" + "=" * 110)
    print("MP vs SUP — SCORE DISTRIBUTION")
    print("=" * 110)
    print(group_summary(df, "pseudo_bucket").to_string(index=False))

    print("\n" + "=" * 110)
    print("CASE1 vs CASE2 — SCORE DISTRIBUTION")
    print("=" * 110)
    print(group_summary(df, "case_type").to_string(index=False))

    r1 = rank1_summary(df)
    if r1:
        print("\n" + "=" * 110)
        print("RANK=1 CANDIDATES — REFERENCE ONLY")
        print("=" * 110)
        for k, v in r1.items():
            if k == "count":
                print(f"{k:<10}: {v:,}")
            else:
                print(f"{k:<10}: {v:.10g}")

    th_path = rdir / "score_threshold_counts.csv"
    pt_path = rdir / "score_percentiles.csv"
    th.to_csv(th_path, index=False)
    pt.to_csv(pt_path, index=False)

    print("\n" + "=" * 110)
    print("SAVED")
    print("=" * 110)
    print(th_path)
    print(pt_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    main(a.cutoff, a.round)
