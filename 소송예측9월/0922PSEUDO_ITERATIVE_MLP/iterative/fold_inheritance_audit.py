import argparse
import pickle
from pathlib import Path

import pandas as pd
import torch

from build_round_data import load_c3_split, pair_set


# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()

parser.add_argument("--from-year", type=int, required=True)
parser.add_argument("--to-year", type=int, required=True)
parser.add_argument("--max-round", type=int, default=10)

parser.add_argument(
    "--snapshot-dir",
    type=str,
    default="../../0916PSEUDO_ONESHOT/snapshots_clean_0917",
)

parser.add_argument(
    "--runs-dir",
    type=str,
    default="../runs",
)

parser.add_argument(
    "--output-dir",
    type=str,
    default="inheritance_audit",
)

args = parser.parse_args()

FROM = args.from_year
TO = args.to_year
MAX_ROUND = args.max_round

SNAP_DIR = Path(args.snapshot_dir)
RUNS_DIR = Path(args.runs_dir)
OUT_DIR = Path(args.output_dir)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def patent_pair(r):
    return (
        str(r["plt_patent"]),
        str(r["def_patent"]),
    )


def get_round_dir(year, r):
    p = (
        RUNS_DIR
        / f"cutoff_{year}"
        / f"round_{r:02d}"
    )

    f = p / "round_data.pkl"

    if not f.exists():
        raise FileNotFoundError(
            f"round_data.pkl 못 찾음: {f}"
        )

    return p


# ============================================================
# Target snapshot
# ============================================================
snap_path = (
    SNAP_DIR
    / f"snapshot_{TO}_v2.pt"
)

print(
    f"\n[LOAD] target snapshot: "
    f"{snap_path}"
)

snap = torch.load(
    snap_path,
    map_location="cpu",
    weights_only=False,
)


# ============================================================
# Target patent / case index
# ============================================================
pat2idx = {
    str(k): int(v)
    for k, v in snap["pat2idx"].items()
}

case2idx = {
    str(k): int(v)
    for k, v in snap["case2idx"].items()
}


# ============================================================
# Filing date map
# ============================================================
# ============================================================
# Filing date map
# source + target snapshot
# ============================================================
filing_map = {}

for year in [FROM, TO]:

    p = SNAP_DIR / f"snapshot_{year}_v2.pt"

    s = torch.load(
        p,
        map_location="cpu",
        weights_only=False,
    )

    pm = s["patent_meta"].copy()

    pm["patent_id"] = (
        pm["patent_id"].astype(str)
    )

    if "filing_date" in pm.columns:
        temp = (
            pm.set_index("patent_id")
            ["filing_date"]
            .to_dict()
        )

        filing_map.update(temp)


# ============================================================
# Target GT MP
# snapshot historical sue
# ============================================================
sue_ei = snap["data"][
    ("patent", "sue", "patent")
].edge_index

gt_mp_raw = set(
    zip(
        sue_ei[0].tolist(),
        sue_ei[1].tolist(),
    )
)


# ============================================================
# Target fixed C3 SUP / VAL
# build_round_data.py와 동일한 split 사용
# ============================================================
(
    _,
    sup_real,
    val_fixed,
    c3_path,
) = load_c3_split(TO)

gt_sup = pair_set(sup_real)
val = pair_set(val_fixed)


# VAL leakage 제거
gt_mp = gt_mp_raw - val


print("\n=== TARGET GT ===")
print(f"C3 split  : {c3_path}")
print(f"GT MP raw : {len(gt_mp_raw):,}")
print(f"GT MP     : {len(gt_mp):,}")
print(f"GT SUP    : {len(gt_sup):,}")
print(f"VAL       : {len(val):,}")


# ============================================================
# Audit
# ============================================================
summary_rows = []
missing_rows = []


for r in range(MAX_ROUND + 1):

    # --------------------------------------------------------
    # Source round_data
    # --------------------------------------------------------
    rd_dir = get_round_dir(FROM, r)

    rd = load_pkl(
        rd_dir / "round_data.pkl"
    )


    # --------------------------------------------------------
    # Source cumulative pseudo
    # MP + SUP
    # --------------------------------------------------------
    pseudo = (
        list(rd.get("mp_pseudo", []))
        + list(rd.get("sup_pseudo", []))
    )


    # --------------------------------------------------------
    # Patent pair 기준 dedup
    # --------------------------------------------------------
    pseudo_dict = {}

    for x in pseudo:
        key = patent_pair(x)
        pseudo_dict[key] = x

    pseudo = list(
        pseudo_dict.values()
    )


    # --------------------------------------------------------
    # Target snapshot remap
    # --------------------------------------------------------
    remapped = set()
    missing = []

    for x in pseudo:

        plt_pat = str(
            x["plt_patent"]
        )

        def_pat = str(
            x["def_patent"]
        )

        plt_alive = (
            plt_pat in pat2idx
        )

        def_alive = (
            def_pat in pat2idx
        )

        if plt_alive and def_alive:

            remapped.add(
                (
                    pat2idx[plt_pat],
                    pat2idx[def_pat],
                )
            )

        else:

            missing.append(
                (
                    x,
                    plt_alive,
                    def_alive,
                )
            )


    # --------------------------------------------------------
    # Target GT collision
    # --------------------------------------------------------
    hit_mp = (
        remapped & gt_mp
    )

    hit_sup = (
        remapped & gt_sup
    )

    hit_val = (
        remapped & val
    )


    # GT와 겹치는 pseudo는
    # inherited pseudo에서 제외
    blocked = (
        hit_mp
        | hit_sup
        | hit_val
    )

    inherited = (
        remapped - blocked
    )


    # --------------------------------------------------------
    # Final target MP
    # --------------------------------------------------------
    final_mp = (
        gt_mp | inherited
    )


    # ========================================================
    # Missing detail
    # ========================================================
    for (
        x,
        plt_alive,
        def_alive,
    ) in missing:

        case_id = str(
            x.get("case_row_id")
        )

        case_alive = (
            case_id in case2idx
        )

        plt_pat = str(
            x["plt_patent"]
        )

        def_pat = str(
            x["def_patent"]
        )


        # --------------------------------------------
        # Missing reason
        # --------------------------------------------
        if not case_alive:

            reason = (
                "CASE_MISSING"
            )

        elif (
            not plt_alive
            and not def_alive
        ):

            reason = (
                "BOTH_PATENT_MISSING"
            )

        elif not plt_alive:

            reason = (
                "PLT_PATENT_MISSING"
            )

        elif not def_alive:

            reason = (
                "DEF_PATENT_MISSING"
            )

        else:

            reason = "UNKNOWN"


        missing_rows.append(
            {
                "Round":
                    f"R{r}",

                "case_row_id":
                    case_id,

                "case_type":
                    x.get("case_type"),

                "case_alive":
                    case_alive,

                "plt_patent":
                    plt_pat,

                "plt_alive":
                    plt_alive,

                "plt_filing_date":
                    filing_map.get(
                        plt_pat
                    ),

                "def_patent":
                    def_pat,

                "def_alive":
                    def_alive,

                "def_filing_date":
                    filing_map.get(
                        def_pat
                    ),

                "reason":
                    reason,
            }
        )


    # ========================================================
    # Summary
    # ========================================================
    summary_rows.append(
        {
            "Round":
                f"R{r}",

            f"Pseudo{str(FROM)[-2:]}":
                len(pseudo),

            f"Remap{str(TO)[-2:]}":
                len(remapped),

            "Missing":
                len(missing),

            "∩GT_MP":
                len(hit_mp),

            "∩GT_SUP":
                len(hit_sup),

            "∩VAL":
                len(hit_val),

            "Inherited_MP":
                len(inherited),

            f"GT_MP{str(TO)[-2:]}":
                len(gt_mp),

            f"Final_MP{str(TO)[-2:]}":
                len(final_mp),
        }
    )


# ============================================================
# DataFrame
# ============================================================
summary = pd.DataFrame(
    summary_rows
)

missing_df = pd.DataFrame(
    missing_rows
)


# ============================================================
# Print
# ============================================================
print(
    "\n=== FOLD INHERITANCE AUDIT ==="
)

print(
    f"{FROM} -> {TO}\n"
)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# Save
# ============================================================
summary_path = (
    OUT_DIR
    / f"audit_{FROM}_to_{TO}.csv"
)

missing_path = (
    OUT_DIR
    / f"missing_{FROM}_to_{TO}.csv"
)


summary.to_csv(
    summary_path,
    index=False,
)

missing_df.to_csv(
    missing_path,
    index=False,
)


print("\nSaved:")
print(summary_path)
print(missing_path)