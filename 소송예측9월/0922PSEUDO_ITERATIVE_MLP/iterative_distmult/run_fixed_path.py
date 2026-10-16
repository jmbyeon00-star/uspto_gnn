from __future__ import annotations
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRAIN = HERE / "train_fixed_round.py"

PATH = [
    (2004, 1),
    (2006, 7),
    (2008, 7),
    (2010, 2),
]

warm = None

for cutoff, round_idx in PATH:
    print("\n" + "=" * 100)
    print(f"FIXED-PATH DistMult | cutoff={cutoff} | MLP-selected R{round_idx}")
    print(f"WARM START = {warm if warm else 'fresh init'}")
    print("=" * 100)

    cmd = [
        sys.executable,
        str(TRAIN),
        "--cutoff", str(cutoff),
        "--round", str(round_idx),
    ]
    if warm is not None:
        cmd += ["--warm-start", str(warm)]

    subprocess.run(cmd, check=True)

    warm = (
        ROOT / "runs_distmult_fixed"
        / f"cutoff_{cutoff}"
        / f"round_{round_idx:02d}"
        / "checkpoint.pt"
    )
    if not warm.exists():
        raise FileNotFoundError(f"Checkpoint not created: {warm}")

print("\n" + "=" * 100)
print("DONE — FIXED-PATH DistMult")
print("=" * 100)
print(f"Final checkpoint: {warm}")
