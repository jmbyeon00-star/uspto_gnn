from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"

THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
PCTS = [.50, .75, .90, .95, .975, .99, .995, .999]


def qstats(s):
    q = s.quantile(PCTS)
    return {
        "count": len(s), "mean": s.mean(), "median": q.loc[.50],
        "p75": q.loc[.75], "p90": q.loc[.90], "p95": q.loc[.95],
        "p97.5": q.loc[.975], "p99": q.loc[.99],
        "p99.5": q.loc[.995], "p99.9": q.loc[.999], "max": s.max()
    }


def main(cutoff, round_idx):
    rdir = RUNS_DIR / f"cutoff_{cutoff}" / f"round_{round_idx:02d}"
    path = rdir / "scored_candidates.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    need = {"case_row_id","known_patent","score","rank","pseudo_bucket","case_type"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"필수 컬럼 없음: {sorted(miss)}")

    # score_case12.py에서 rank는 (case_row_id, known_patent) query 내부 순위.
    # 따라서 rank=1 하나가 missing-patent query 하나의 Top1.
    top1 = df[df["rank"].eq(1)].copy()

    # Top2를 같은 query에서 붙여 margin 계산.
    top2 = (
        df[df["rank"].eq(2)]
        [["case_row_id","known_patent","score","candidate_patent"]]
        .rename(columns={"score":"top2_score","candidate_patent":"top2_patent"})
    )
    top1 = top1.merge(top2, on=["case_row_id","known_patent"], how="left")
    top1 = top1.rename(columns={
        "score":"top1_score",
        "candidate_patent":"top1_patent"
    })
    top1["margin"] = top1["top1_score"] - top1["top2_score"]

    rows = []
    for t in THRESHOLDS:
        x = top1[top1["top1_score"] >= t]
        rows.append({
            "threshold": t,
            "ALL": len(x),
            "ALL_pct": 100 * len(x) / len(top1) if len(top1) else 0,
            "MP": int(x["pseudo_bucket"].eq("mp").sum()),
            "SUP": int(x["pseudo_bucket"].eq("sup").sum()),
            "CASE1": int(x["case_type"].eq("case1").sum()),
            "CASE2": int(x["case_type"].eq("case2").sum()),
        })
    th = pd.DataFrame(rows)

    print("=" * 110)
    print(f"TOP1 PSEUDO-CANDIDATE ANALYSIS | cutoff={cutoff} | round={round_idx}")
    print("=" * 110)
    print(f"all candidate rows : {len(df):,}")
    print(f"Top1 queries       : {len(top1):,}")
    print(f"Top2 available     : {top1['top2_score'].notna().sum():,}")
    print(f"Top2 missing       : {top1['top2_score'].isna().sum():,}")

    print("\n" + "=" * 110)
    print("TOP1 ABSOLUTE SIGMOID THRESHOLDS")
    print("=" * 110)
    print(th.to_string(
        index=False,
        formatters={
            "threshold": lambda x: f">= {x:.2f}",
            "ALL_pct": lambda x: f"{x:.4f}%"
        }
    ))

    print("\n" + "=" * 110)
    print("TOP1 SCORE DISTRIBUTION")
    print("=" * 110)
    for k, v in qstats(top1["top1_score"]).items():
        print(f"{k:<10}: {v:,}" if k == "count" else f"{k:<10}: {v:.10g}")

    m = top1["margin"].dropna()
    print("\n" + "=" * 110)
    print("TOP1 - TOP2 MARGIN DISTRIBUTION | REFERENCE ONLY")
    print("=" * 110)
    if len(m):
        for k, v in qstats(m).items():
            print(f"{k:<10}: {v:,}" if k == "count" else f"{k:<10}: {v:.10g}")
    else:
        print("No Top2 scores.")

    print("\n" + "=" * 110)
    print("TOP1 SCORE BY MP / SUP")
    print("=" * 110)
    for name, g in top1.groupby("pseudo_bucket"):
        s = g["top1_score"]
        print(f"{name:<5} n={len(g):,} median={s.median():.10g} "
              f"p95={s.quantile(.95):.10g} p99={s.quantile(.99):.10g} max={s.max():.10g}")

    print("\n" + "=" * 110)
    print("TOP1 SCORE BY CASE TYPE")
    print("=" * 110)
    for name, g in top1.groupby("case_type"):
        s = g["top1_score"]
        print(f"{name:<7} n={len(g):,} median={s.median():.10g} "
              f"p95={s.quantile(.95):.10g} p99={s.quantile(.99):.10g} max={s.max():.10g}")

    # 다음 select_pseudo.py에서 그대로 사용할 수 있도록 Top1 요약도 보존.
    top1_path = rdir / "top1_candidates.parquet"
    th_path = rdir / "top1_threshold_counts.csv"
    top1.to_parquet(top1_path, index=False)
    th.to_csv(th_path, index=False)

    print("\n" + "=" * 110)
    print("SAVED")
    print("=" * 110)
    print(top1_path)
    print(th_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    main(a.cutoff, a.round)
