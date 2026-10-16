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

args = parser.parse_args()

FROM = args.from_year
TO = args.to_year
MAX_ROUND = args.max_round

SNAP_DIR = Path(args.snapshot_dir)
RUNS_DIR = Path(args.runs_dir)

SRC_ROOT = RUNS_DIR / f"cutoff_{FROM}"
OUT_ROOT = (
    RUNS_DIR
    / f"predecessor_{FROM}_to_{TO}"
)

OUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def patent_pair(record):
    return (
        str(record["plt_patent"]),
        str(record["def_patent"]),
    )


def collect_source_pseudo(round_data):
    """
    현재 source round가 보유한 전체 pseudo 계보.

    최초 fold(2004):
      mp_inherited 없음
      + mp_pseudo
      + sup_pseudo

    다음 fold(2006+):
      이전 fold에서 넘어온 mp_inherited
      + 현재 fold mp_pseudo
      + 현재 fold sup_pseudo

    patent pair 기준 dedup.
    """
    records = []

    records += list(
        round_data.get("mp_inherited", [])
    )
    records += list(
        round_data.get("mp_pseudo", [])
    )
    records += list(
        round_data.get("sup_pseudo", [])
    )

    dedup = {}

    for x in records:
        dedup[patent_pair(x)] = x

    return list(dedup.values())


def get_round_data(round_idx):
    round_dir = (
        SRC_ROOT
        / f"round_{round_idx:02d}"
    )

    path = round_dir / "round_data.pkl"

    if not path.exists():
        raise FileNotFoundError(
            f"round_data.pkl 못 찾음: {path}"
        )

    return (
        round_dir,
        load_pkl(path),
    )


# ============================================================
# Target snapshot
# ============================================================
snap_path = (
    SNAP_DIR
    / f"snapshot_{TO}_v2.pt"
)

print(f"\n[LOAD] {snap_path}")

snap = torch.load(
    snap_path,
    map_location="cpu",
    weights_only=False,
)

pat2idx = {str(k): int(v) for k, v in snap["pat2idx"].items()}
idx2pat = {int(v): str(k) for k, v in pat2idx.items()}


# ============================================================
# Target GT MP
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
# Target fixed SUP / VAL
# ============================================================
(
    _,
    sup_real,
    val_fixed,
    c3_path,
) = load_c3_split(TO)

gt_sup_idx = pair_set(sup_real)
val_idx = pair_set(val_fixed)

def idx_pair_to_patent_pair(p):
    return idx2pat[int(p[0])], idx2pat[int(p[1])]

gt_mp_pat = {idx_pair_to_patent_pair(p) for p in gt_mp_raw}
gt_sup_pat = {idx_pair_to_patent_pair(p) for p in gt_sup_idx}
val_pat = {idx_pair_to_patent_pair(p) for p in val_idx}

# fixed VAL은 historical MP에서도 제거. Identity 비교는 patent ID 기준.
gt_mp_pat = gt_mp_pat - val_pat
gt_mp = {(pat2idx[a], pat2idx[b]) for a, b in gt_mp_pat}
gt_sup = {(pat2idx[a], pat2idx[b]) for a, b in gt_sup_pat}
val = {(pat2idx[a], pat2idx[b]) for a, b in val_pat}


print("\n=== TARGET ===")
print(f"C3 split : {c3_path}")
print(f"GT MP    : {len(gt_mp):,}")
print(f"GT SUP   : {len(gt_sup):,}")
print(f"VAL      : {len(val):,}")


# ============================================================
# Build inherited.pkl for every source round
# ============================================================
summary_rows = []

for r in range(MAX_ROUND + 1):

    print(
        "\n" + "=" * 70
    )
    print(
        f"[SOURCE R{r}] "
        f"{FROM} -> {TO}"
    )
    print(
        "=" * 70
    )

    round_dir, rd = get_round_data(r)

    # --------------------------------------------------------
    # 전체 pseudo lineage
    # --------------------------------------------------------
    pseudo = collect_source_pseudo(rd)

    # --------------------------------------------------------
    # Target snapshot으로 remap
    #
    # remap_dict:
    #   target index pair -> original pseudo record
    #
    # original patent ID record를 같이 보존해야
    # 다음 fold에서도 다시 remap 가능.
    # --------------------------------------------------------
    remap_dict = {}
    missing_records = []

    for x in pseudo:
        pp = patent_pair(x)
        plt_pat, def_pat = pp
        plt_alive = plt_pat in pat2idx
        def_alive = def_pat in pat2idx

        if not (plt_alive and def_alive):
            missing_records.append({
                "record": x,
                "plt_alive": plt_alive,
                "def_alive": def_alive,
            })
            continue

        # 영구 identity는 patent ID pair. index는 target snapshot용 좌표일 뿐.
        remap_dict[pp] = x

    remapped_pat = set(remap_dict)

    hit_mp_pat = remapped_pat & gt_mp_pat
    hit_sup_pat = remapped_pat & gt_sup_pat
    hit_val_pat = remapped_pat & val_pat
    blocked_pat = hit_mp_pat | hit_sup_pat | hit_val_pat
    inherited_pat = remapped_pat - blocked_pat

    inherited_patent_pairs = sorted(inherited_pat)
    inherited_pairs = [(pat2idx[a], pat2idx[b]) for a, b in inherited_patent_pairs]

    inherited_records = []
    for pp, target_pair in zip(inherited_patent_pairs, inherited_pairs):
        x = dict(remap_dict[pp])
        if "src" in x: x.setdefault("source_src", x["src"])
        if "dst" in x: x.setdefault("source_dst", x["dst"])
        x["src"], x["dst"] = int(target_pair[0]), int(target_pair[1])
        x["plt_patent"], x["def_patent"] = pp
        inherited_records.append(x)

    remapped = {(pat2idx[a], pat2idx[b]) for a, b in remapped_pat}
    hit_mp = {(pat2idx[a], pat2idx[b]) for a, b in hit_mp_pat}
    hit_sup = {(pat2idx[a], pat2idx[b]) for a, b in hit_sup_pat}
    hit_val = {(pat2idx[a], pat2idx[b]) for a, b in hit_val_pat}
    inherited = set(inherited_pairs)

    final_mp = (
        gt_mp | inherited
    )

    # --------------------------------------------------------
    # Save directory
    # --------------------------------------------------------
    out_dir = (
        OUT_ROOT
        / f"source_round_{r:02d}"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_ckpt = (
        round_dir / "checkpoint.pt"
    )

    # checkpoint는 여기서 복사하지 않음.
    # 다음 train_predecessor 단계에서
    # source checkpoint로 warm-start한다.
    # --------------------------------------------------------
    payload = {
        "from_year": FROM,
        "to_year": TO,
        "source_round": r,

        "source_round_dir":
            str(round_dir),

        "source_checkpoint":
            str(source_ckpt),

        # 다음 fold 전달용:
        # patent ID / provenance 보존
        "inherited_records":
            inherited_records,

        # 현재 target snapshot에서
        # 바로 edge_index로 쓸 수 있는 pair
        "inherited_pairs":
            inherited_pairs,

        # 영구 identity / cross-snapshot 중복검사용
        "inherited_patent_pairs":
            inherited_patent_pairs,

        # audit / 검증용
        "missing_records":
            missing_records,

        "blocked_pairs": {
            "gt_mp": sorted(hit_mp),
            "gt_sup": sorted(hit_sup),
            "val": sorted(hit_val),
        },

        "stats": {
            "source_pseudo":
                len(pseudo),

            "remapped":
                len(remapped),

            "missing":
                len(missing_records),

            "hit_gt_mp":
                len(hit_mp),

            "hit_gt_sup":
                len(hit_sup),

            "hit_val":
                len(hit_val),

            "inherited":
                len(inherited),

            "gt_mp":
                len(gt_mp),

            "gt_sup":
                len(gt_sup),

            "val":
                len(val),

            "final_mp":
                len(final_mp),
        },
    }

    out_path = (
        out_dir / "inherited.pkl"
    )

    save_pkl(
        payload,
        out_path,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    summary_rows.append(
        {
            "source_round": f"R{r}",
            "source_pseudo":
                len(pseudo),
            "remapped":
                len(remapped),
            "missing":
                len(missing_records),
            "hit_gt_mp":
                len(hit_mp),
            "hit_gt_sup":
                len(hit_sup),
            "hit_val":
                len(hit_val),
            "inherited_mp":
                len(inherited),
            "gt_mp":
                len(gt_mp),
            "final_mp":
                len(final_mp),
        }
    )

    print(
        f"source pseudo : {len(pseudo):,}"
    )
    print(
        f"remapped      : {len(remapped):,}"
    )
    print(
        f"missing       : {len(missing_records):,}"
    )
    print(
        f"hit GT_MP     : {len(hit_mp):,}"
    )
    print(
        f"hit GT_SUP    : {len(hit_sup):,}"
    )
    print(
        f"hit VAL       : {len(hit_val):,}"
    )
    print(
        f"inherited MP  : {len(inherited):,}"
    )
    print(
        f"final MP      : {len(final_mp):,}"
    )
    print(
        f"saved         : {out_path}"
    )


# ============================================================
# Save summary
# ============================================================
summary = pd.DataFrame(
    summary_rows
)

summary_path = (
    OUT_ROOT
    / "inheritance_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

print(
    "\n=== INHERITANCE SUMMARY ==="
)

print(
    summary.to_string(
        index=False
    )
)

print(
    f"\nSaved: {summary_path}"
)
