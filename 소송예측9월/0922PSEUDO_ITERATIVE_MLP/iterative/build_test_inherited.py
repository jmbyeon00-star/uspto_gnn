import pickle
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]

SOURCE_CUTOFF = 2010
SOURCE_ROUND = 2
TEST_CUTOFF = 2012

RUNS_DIR = ROOT / "runs"

SNAP_DIR = (
    ROOT.parent
    / "0916PSEUDO_ONESHOT"
    / "snapshots_clean_0917"
)

ROUND_DIR = (
    RUNS_DIR
    / f"cutoff_{SOURCE_CUTOFF}"
    / f"round_{SOURCE_ROUND:02d}"
)

TEST_GT_PATH = (
    ROOT
    / "fixed_data"
    / "test"
    / "test_gt_2013_2014.pkl"
)

OUT_DIR = (
    RUNS_DIR
    / "final_test_2012"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def patent_pair(x):
    return (
        str(x["plt_patent"]),
        str(x["def_patent"]),
    )


# ============================================================
# 1. 2010 R2 전체 pseudo lineage
# ============================================================

rd = load_pkl(
    ROUND_DIR / "round_data.pkl"
)

records = []

# 혹시 별도 inherited key가 있는 경우 포함
records += list(
    rd.get("mp_inherited", [])
)

records += list(
    rd.get("mp_pseudo", [])
)

records += list(
    rd.get("sup_pseudo", [])
)

# patent pair 기준 dedup
pseudo_dict = {}

for x in records:
    pseudo_dict[patent_pair(x)] = x

source_pseudo = list(
    pseudo_dict.values()
)

print("=" * 70)
print("SOURCE")
print("=" * 70)

print(
    "2010 R2 mp_pseudo :",
    len(rd.get("mp_pseudo", [])),
)

print(
    "2010 R2 sup_pseudo:",
    len(rd.get("sup_pseudo", [])),
)

print(
    "전체 pseudo lineage:",
    len(source_pseudo),
)


# ============================================================
# 2. 2012 snapshot
# ============================================================

snap_path = (
    SNAP_DIR
    / "snapshot_2012_v2.pt"
)

snap = torch.load(
    snap_path,
    map_location="cpu",
    weights_only=False,
)

pat2idx = {
    str(k): int(v)
    for k, v in snap["pat2idx"].items()
}

idx2pat = {
    int(v): str(k)
    for k, v in pat2idx.items()
}


# ============================================================
# 3. 2012 실제 historical sue
# ============================================================

sue_ei = snap["data"][
    ("patent", "sue", "patent")
].edge_index

gt_mp_idx = set(
    zip(
        sue_ei[0].tolist(),
        sue_ei[1].tolist(),
    )
)

gt_mp_pat = {
    (
        idx2pat[int(s)],
        idx2pat[int(d)],
    )
    for s, d in gt_mp_idx
}

print()
print("=" * 70)
print("2012 SNAPSHOT")
print("=" * 70)

print(
    "2012 historical GT MP:",
    len(gt_mp_pat),
)


# ============================================================
# 4. 2010 R2 pseudo -> 2012 remap
# ============================================================

remapped = {}
missing_records = []

for x in source_pseudo:

    pp = patent_pair(x)

    a, b = pp

    a_alive = a in pat2idx
    b_alive = b in pat2idx

    if not (
        a_alive and b_alive
    ):
        missing_records.append({
            "record": x,
            "plt_alive": a_alive,
            "def_alive": b_alive,
        })
        continue

    remapped[pp] = x


remapped_pat = set(remapped)

# 2012 시점에 이미 실제 소송이 된 pair
hit_gt_mp = (
    remapped_pat
    & gt_mp_pat
)

# 이미 실제 edge면 pseudo로 중복 추가하지 않음
inherited_pat = (
    remapped_pat
    - hit_gt_mp
)


# ============================================================
# 5. Test GT 104개와 leakage 재확인
# ============================================================

test_obj = load_pkl(
    TEST_GT_PATH
)

test_gt = {
    (str(a), str(b))
    for a, b
    in test_obj["test_patent_pairs"]
}

hit_test = (
    inherited_pat
    & test_gt
)


# ============================================================
# 6. 2012 index 기준 inherited records
# ============================================================

inherited_patent_pairs = sorted(
    inherited_pat
)

inherited_pairs = [
    (
        pat2idx[a],
        pat2idx[b],
    )
    for a, b
    in inherited_patent_pairs
]

inherited_records = []

for pp, pair in zip(
    inherited_patent_pairs,
    inherited_pairs,
):

    x = dict(remapped[pp])

    if "src" in x:
        x.setdefault(
            "source_src",
            x["src"],
        )

    if "dst" in x:
        x.setdefault(
            "source_dst",
            x["dst"],
        )

    x["src"] = int(pair[0])
    x["dst"] = int(pair[1])

    x["plt_patent"] = pp[0]
    x["def_patent"] = pp[1]

    x["inherited"] = True

    inherited_records.append(x)


# ============================================================
# 7. 결과
# ============================================================

print()
print("=" * 70)
print("2010 R2 -> 2012 TEST REMAP")
print("=" * 70)

print(
    "Source pseudo     :",
    len(source_pseudo),
)

print(
    "2012 remapped     :",
    len(remapped_pat),
)

print(
    "Missing           :",
    len(missing_records),
)

print(
    "2012 GT MP overlap:",
    len(hit_gt_mp),
)

print(
    "Final inherited   :",
    len(inherited_pat),
)

print(
    "Test GT           :",
    len(test_gt),
)

print(
    "Test GT overlap   :",
    len(hit_test),
)


# ============================================================
# 8. 저장
# ============================================================

payload = {
    "from_year":
        SOURCE_CUTOFF,

    "to_year":
        TEST_CUTOFF,

    "source_round":
        SOURCE_ROUND,

    "source_checkpoint":
        str(
            ROUND_DIR
            / "checkpoint.pt"
        ),

    "inherited_records":
        inherited_records,

    "inherited_pairs":
        inherited_pairs,

    "inherited_patent_pairs":
        inherited_patent_pairs,

    "missing_records":
        missing_records,

    "blocked_gt_mp_patent_pairs":
        sorted(hit_gt_mp),

    "test_overlap_patent_pairs":
        sorted(hit_test),

    "stats": {
        "source_pseudo":
            len(source_pseudo),

        "remapped":
            len(remapped_pat),

        "missing":
            len(missing_records),

        "hit_gt_mp":
            len(hit_gt_mp),

        "inherited":
            len(inherited_pat),

        "test_gt":
            len(test_gt),

        "hit_test":
            len(hit_test),
    },
}

out_path = (
    OUT_DIR
    / "inherited_from_2010_R2.pkl"
)

with open(
    out_path,
    "wb",
) as f:
    pickle.dump(
        payload,
        f,
    )

print()
print("saved:", out_path)

if hit_test:
    print()
    print("WARNING: Test GT overlap 존재")
    for p in sorted(hit_test):
        print(" ", p)

