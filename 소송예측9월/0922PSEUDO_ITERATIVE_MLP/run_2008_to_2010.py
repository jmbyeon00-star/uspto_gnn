import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

THRESHOLD = 0.99


def run(cmd):
    print("\n" + "=" * 100)
    print("RUN:", " ".join(map(str, cmd)))
    print("=" * 100, flush=True)

    subprocess.run(
        [str(x) for x in cmd],
        cwd=ROOT,
        check=True,
    )


# ============================================================
# PART 0. SAFETY CHECK
# 공식 2008 R0 checkpoint가 존재하는지 확인
# ============================================================

official_2008_r0 = ROOT / "runs/cutoff_2008/round_00/checkpoint.pt"

if not official_2008_r0.exists():
    raise FileNotFoundError(
        f"\n공식 2008 R0 checkpoint가 없습니다:\n{official_2008_r0}\n"
        "먼저 2006 predecessor winner를 공식 2008 R0으로 복사하세요."
    )

print("\nOFFICIAL 2008 R0 FOUND")
print(official_2008_r0)


# ============================================================
# PART 1. 2008 R0 -> R10 iterative pseudo training
# ============================================================

print("\n")
print("#" * 100)
print("# PART 1: 2008 R0 -> R10")
print("#" * 100)


for r in range(10):

    next_r = r + 1

    print("\n")
    print("=" * 100)
    print(f"2008 ROUND {r} -> ROUND {next_r}")
    print("=" * 100)

    # --------------------------------------------------------
    # 1. CASE1/2 scoring
    # --------------------------------------------------------

    run([
        PYTHON,
        "iterative/score_case12.py",
        "--cutoff", "2008",
        "--target-start", "2009",
        "--target-end", "2010",
        "--round", str(r),
    ])

    # --------------------------------------------------------
    # 2. Top1 analysis
    # --------------------------------------------------------

    run([
        PYTHON,
        "iterative/analyze_top1.py",
        "--cutoff", "2008",
        "--round", str(r),
    ])

    # --------------------------------------------------------
    # 3. Pseudo selection
    # --------------------------------------------------------

    run([
        PYTHON,
        "iterative/select_pseudo.py",
        "--cutoff", "2008",
        "--target-start", "2009",
        "--target-end", "2010",
        "--round", str(r),
        "--threshold", str(THRESHOLD),
    ])

    # --------------------------------------------------------
    # 4. Build next round
    # --------------------------------------------------------

    run([
        PYTHON,
        "iterative/build_round_data.py",
        "--cutoff", "2008",
        "--target-start", "2009",
        "--target-end", "2010",
        "--round", str(next_r),
    ])

    # --------------------------------------------------------
    # 5. Train next round
    # --------------------------------------------------------

    run([
        PYTHON,
        "iterative/train_round.py",
        "--cutoff", "2008",
        "--round", str(next_r),
    ])


print("\n")
print("#" * 100)
print("# 2008 R0 -> R10 COMPLETE")
print("#" * 100)


# ============================================================
# PART 2. Build clean 2010 R0
# ============================================================

print("\n")
print("#" * 100)
print("# PART 2: BUILD CLEAN 2010 R0")
print("#" * 100)


run([
    PYTHON,
    "iterative/build_round_data.py",
    "--cutoff", "2010",
    "--target-start", "2011",
    "--target-end", "2012",
    "--round", "0",
])


# ============================================================
# PART 3. 2008 R0~R10 -> 2010 R0 predecessor comparison
# ============================================================

print("\n")
print("#" * 100)
print("# PART 3: 2008 R0~R10 -> 2010 R0")
print("#" * 100)


for source_round in range(11):

    print("\n")
    print("=" * 100)
    print(f"2008 R{source_round} -> 2010 R0")
    print("=" * 100)

    source_ckpt = (
        ROOT
        / f"runs/cutoff_2008/round_{source_round:02d}/checkpoint.pt"
    )

    if not source_ckpt.exists():
        raise FileNotFoundError(
            f"2008 R{source_round} checkpoint 없음: {source_ckpt}"
        )

    run([
        PYTHON,
        "iterative/train_predecessor.py",
        "--cutoff", "2010",
        "--round", "0",
        "--source-cutoff", "2008",
        "--source-round", str(source_round),
    ])


# ============================================================
# PART 4. Compare 2010 fixed VAL MRR
# ============================================================

print("\n")
print("#" * 100)
print("# PART 4: 2010 PREDECESSOR RESULTS")
print("#" * 100)


base = ROOT / "runs/cutoff_2010/predecessor_test"

rows = []

for r in range(11):

    metrics_path = (
        base
        / f"from_2008_round_{r:02d}"
        / "metrics.json"
    )

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"metrics.json 없음: {metrics_path}"
        )

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    rows.append({
        "source_round": r,
        "best_epoch": metrics["best_epoch"],
        "mrr": float(metrics["best_val_MRR"]),
        "checkpoint": (
            base
            / f"from_2008_round_{r:02d}"
            / "checkpoint.pt"
        ),
    })


rows.sort(
    key=lambda x: x["mrr"],
    reverse=True,
)


print("\n")
print("=" * 70)
print("2008 R0~R10 -> 2010 R0 PREDECESSOR RESULTS")
print("=" * 70)

print(
    f"{'Rank':<7}"
    f"{'Source':<15}"
    f"{'Epoch':>10}"
    f"{'2010 VAL MRR':>20}"
)

print("-" * 70)


for rank, row in enumerate(rows, 1):

    print(
        f"{rank:<7}"
        f"{('2008 R' + str(row['source_round'])):<15}"
        f"{row['best_epoch']:>10}"
        f"{row['mrr']:>20.9f}"
    )


# ============================================================
# BEST
# ============================================================

best = rows[0]

print("\n")
print("=" * 100)
print("BEST PREDECESSOR")
print("=" * 100)

print(
    f"""
BEST SOURCE ROUND : 2008 R{best['source_round']}
BEST EPOCH        : {best['best_epoch']}
2010 VAL MRR      : {best['mrr']:.9f}

CHECKPOINT:
{best['checkpoint']}

IMPORTANT:
아직 이 checkpoint를 공식 2010 R0으로 복사하지 않았습니다.
결과를 확인한 뒤 공식 2010 R0을 결정하세요.

STOPPED BEFORE 2010 R1~R10.
"""
)

print("=" * 100)
