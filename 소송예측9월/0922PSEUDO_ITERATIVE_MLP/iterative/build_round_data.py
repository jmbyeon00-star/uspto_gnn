from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FIXED_DIR = ROOT / "fixed_data"
POOL_DIR = FIXED_DIR / "unified_pools"
C3_DIR = FIXED_DIR / "c3_pools"

RUNS_DIR = ROOT / "runs"


# ============================================================
# BASIC UTILS
# ============================================================

def load_pkl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def pair_set(records):
    """
    list[dict] -> {(src, dst), ...}

    record:
        {
            "src": ...,
            "dst": ...
        }
    """
    out = set()

    for r in records:
        out.add((int(r["src"]), int(r["dst"])))

    return out


def dedup_records(records):
    """
    동일 (src,dst)가 여러 번 있으면 하나만 유지.
    최초 record 유지.
    """
    seen = set()
    out = []

    for r in records:
        pair = (int(r["src"]), int(r["dst"]))

        if pair in seen:
            continue

        seen.add(pair)
        out.append(r)

    return out


# ============================================================
# C3 FIXED SPLIT
# ============================================================
def find_c3_file(cutoff: int):
    """
    Iterative 학습용으로 확정한 C3 7:3 split만 사용.
    """
    path = C3_DIR / f"cutoff_{cutoff}" / "c3_data.pkl"

    if not path.exists():
        raise FileNotFoundError(
            f"\nC3 7:3 fixed split을 찾을 수 없습니다.\n"
            f"cutoff={cutoff}\n"
            f"path={path}"
        )

    return path



def load_c3_split(cutoff: int):
    path = find_c3_file(cutoff)
    obj = load_pkl(path)

    if "train_records" in obj:
        sup_real = obj["train_records"]

    elif "sup_pairs" in obj:
        sup_real = obj["sup_pairs"]

    else:
        raise KeyError(
            f"{path}에서 C3 SUP를 찾지 못했습니다.\n"
            f"keys={list(obj.keys())}"
        )

    if "val_records" in obj:
        val_fixed = obj["val_records"]

    elif "val_pairs" in obj:
        val_fixed = obj["val_pairs"]

    else:
        raise KeyError(
            f"{path}에서 C3 VAL을 찾지 못했습니다.\n"
            f"keys={list(obj.keys())}"
        )

    return obj, sup_real, val_fixed, path


# ============================================================
# UNIFIED NEGATIVE POOL
# ============================================================

def load_unified_pool(cutoff: int):
    path = POOL_DIR / f"pool_{cutoff}.pkl"
    obj = load_pkl(path)

    if "pools" not in obj:
        raise KeyError(
            f"{path}에 'pools'가 없습니다.\n"
            f"keys={list(obj.keys())}"
        )

    pools = {
        int(src): {int(dst) for dst in dsts}
        for src, dsts in obj["pools"].items()
    }

    real_positive_by_src = {
        int(src): {int(dst) for dst in dsts}
        for src, dsts in obj.get(
            "real_positive_by_src", {}
        ).items()
    }

    return obj, pools, real_positive_by_src, path


# ============================================================
# PREVIOUS PSEUDO
# ============================================================
def load_previous_pseudo(cutoff: int, round_idx: int):
    """
    이전 round에서 선택된 pseudo를 누적해서 읽는다.

    round_idx=0
        -> 이전 pseudo 없음

    round_idx=1
        -> round_00/pseudo_selected_for_round_01.pkl

    round_idx=2
        -> round_00/pseudo_selected_for_round_01.pkl
         + round_01/pseudo_selected_for_round_02.pkl

    이후 split_pseudo_by_time()에서 date_filed 기준으로
    MP / SUP를 다시 분리한다.
    """

    if round_idx == 0:
        return []

    records = []

    for r in range(round_idx):
        next_round = r + 1

        path = (
            RUNS_DIR
            / f"cutoff_{cutoff}"
            / f"round_{r:02d}"
            / f"pseudo_selected_for_round_{next_round:02d}.pkl"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Round {round_idx}를 만들기 위해 필요한 "
                f"pseudo 파일이 없습니다:\n{path}"
            )

        with open(path, "rb") as f:
            obj = pickle.load(f)

        # --------------------------------------------------
        # 파일 자체가 올바른 round/cutoff인지 검증
        # --------------------------------------------------

        if int(obj["cutoff"]) != cutoff:
            raise ValueError(
                f"cutoff mismatch: {obj['cutoff']} != {cutoff}\n"
                f"path={path}"
            )

        if int(obj["source_round"]) != r:
            raise ValueError(
                f"source_round mismatch: "
                f"{obj['source_round']} != {r}\n"
                f"path={path}"
            )

        if int(obj["next_round"]) != next_round:
            raise ValueError(
                f"next_round mismatch: "
                f"{obj['next_round']} != {next_round}\n"
                f"path={path}"
            )

        # select_pseudo.py에서 확정된 MP + SUP를 모두 읽음.
        # 여기서는 합쳐서 읽고,
        # 아래 split_pseudo_by_time()에서 시간 기준으로 다시 분리.
        current_records = (
            obj.get("pseudo_mp", [])
            + obj.get("pseudo_sup", [])
        )

        for row in current_records:
            row = dict(row)

            required = {
                "case_row_id",
                "date_filed",
                "src",
                "dst",
            }

            missing = required - set(row.keys())

            if missing:
                raise KeyError(
                    f"{path} pseudo record에 필요한 컬럼이 없습니다: "
                    f"{missing}"
                )

            row["date_filed"] = pd.to_datetime(
                row["date_filed"],
                errors="coerce",
            )

            if pd.isna(row["date_filed"]):
                raise ValueError(
                    f"{path}에 date_filed 파싱 실패가 있습니다.\n"
                    f"case_row_id={row.get('case_row_id')}"
                )

            row["src"] = int(row["src"])
            row["dst"] = int(row["dst"])

            # 어느 selection round에서 들어왔는지 기록
            row["selected_round"] = r

            records.append(row)

    return records


# ============================================================
# INHERITED PSEUDO FROM SELECTED PREDECESSOR
# ============================================================
def load_inherited_pseudo(cutoff: int, allow_empty: bool = False):
    """
    Load pseudo edges inherited from the predecessor fold selected by
    predecessor validation.

    cutoff=2004 has no predecessor -> []
    cutoff>2004 expects:
      runs/predecessor_{cutoff-2}_to_{cutoff}/selected.pkl

    selected.pkl determines the selected source round. The corresponding
    source_round_XX/inherited.pkl already contains src/dst remapped to the
    current cutoff snapshot.
    """
    if cutoff <= 2004:
        return [], None, None

    from_year = cutoff - 2
    pred_dir = RUNS_DIR / f"predecessor_{from_year}_to_{cutoff}"
    selected_path = pred_dir / "selected.pkl"

    if not selected_path.exists():
        if allow_empty:
            print(
                "[PREDECESSOR BASE] selected.pkl 없음 "
                "-> inherited pseudo 없이 clean base 생성"
            )
            return [], None, None

        raise FileNotFoundError(
            f"Selected predecessor file not found:\n{selected_path}"
        )

    selected = load_pkl(selected_path)

    source_round = selected.get("source_round")
    if source_round is None:
        source_round = selected.get("selected_source_round")
    if source_round is None:
        source_round = selected.get("selected_round")
    if source_round is None:
        raise KeyError(
            f"Cannot determine selected source round from {selected_path}. "
            f"keys={list(selected.keys())}"
        )

    source_round = int(source_round)
    inherited_path = (
        pred_dir / f"source_round_{source_round:02d}" / "inherited.pkl"
    )

    if not inherited_path.exists():
        raise FileNotFoundError(
            f"Inherited pseudo file not found:\n{inherited_path}"
        )

    inherited_obj = load_pkl(inherited_path)

    if int(inherited_obj.get("to_year", cutoff)) != cutoff:
        raise ValueError(
            f"Inherited target mismatch: "
            f"{inherited_obj.get('to_year')} != {cutoff}\n"
            f"path={inherited_path}"
        )

    records = []
    for row in inherited_obj.get("inherited_records", []):
        row = dict(row)
        required = {"case_row_id", "date_filed", "src", "dst"}
        missing = required - set(row.keys())
        if missing:
            raise KeyError(
                f"{inherited_path} inherited record missing: {missing}"
            )

        row["date_filed"] = pd.to_datetime(row["date_filed"], errors="coerce")
        if pd.isna(row["date_filed"]):
            raise ValueError(
                f"date_filed parse failed in {inherited_path}: "
                f"case_row_id={row.get('case_row_id')}"
            )

        row["src"] = int(row["src"])
        row["dst"] = int(row["dst"])
        row["inherited"] = True
        row["inherited_from_cutoff"] = from_year
        row["inherited_source_round"] = source_round
        records.append(row)

    return records, inherited_path, selected_path


# ============================================================
# SPLIT PSEUDO -> MP / SUP
# ============================================================

def split_pseudo_by_time(
    records,
    cutoff: int,
    target_start: int,
    target_end: int,
):
    """
    <= cutoff
        -> MP pseudo

    target_start ~ target_end
        -> SUP pseudo

    그 외는 현재 fold에서 사용하지 않음.
    """

    mp = []
    sup = []
    outside = []

    for r in records:
        dt = pd.Timestamp(r["date_filed"])
        year = int(dt.year)

        if year <= cutoff:
            mp.append(r)

        elif target_start <= year <= target_end:
            sup.append(r)

        else:
            outside.append(r)

    return mp, sup, outside


# ============================================================
# CONFLICT / DUPLICATE CONTROL
# ============================================================

def clean_pseudo(
    mp_pseudo,
    sup_pseudo,
    sup_real,
    val_fixed,
):
    """
    우선순위:

    VAL 보호
    real SUP 보호

    pseudo 내부에서는:
        MP 우선
        SUP에서 MP와 겹치는 pair 제거

    이유:
    동일 pair가 과거 pseudo MP로 이미 graph에 들어간다면
    미래 pseudo SUP로 동시에 사용하지 않도록 함.
    """

    val_pairs = pair_set(val_fixed)
    real_sup_pairs = pair_set(sup_real)

    mp_pseudo = dedup_records(mp_pseudo)
    sup_pseudo = dedup_records(sup_pseudo)

    # VAL과 겹치는 pseudo 제거
    mp_pseudo = [
        r for r in mp_pseudo
        if (int(r["src"]), int(r["dst"]))
        not in val_pairs
    ]

    sup_pseudo = [
        r for r in sup_pseudo
        if (int(r["src"]), int(r["dst"]))
        not in val_pairs
    ]

    # 이미 real SUP이면 pseudo SUP로 중복 학습할 필요 없음
    sup_pseudo = [
        r for r in sup_pseudo
        if (int(r["src"]), int(r["dst"]))
        not in real_sup_pairs
    ]

    # MP ↔ SUP pseudo 중복 제거
    mp_pairs = pair_set(mp_pseudo)

    sup_pseudo = [
        r for r in sup_pseudo
        if (int(r["src"]), int(r["dst"]))
        not in mp_pairs
    ]

    return mp_pseudo, sup_pseudo


# ============================================================
# ACTIVE NEGATIVE POOL
# ============================================================
def build_active_negative_pool(
    unified_pools,
    real_positive_by_src,
    sup_real,
    val_fixed,
    val_reserved_pairs,
    mp_pseudo,
    sup_pseudo,
):

    """
    unified hard-negative reservoir에서
    현재 positive로 간주되는 pair를 모두 제거.
    """

    positive_by_src = {}

    def add_positive(src, dst):
        src = int(src)
        dst = int(dst)

        positive_by_src.setdefault(
            src, set()
        ).add(dst)

    # snapshot/history에서 이미 알려진 real positive
    for src, dsts in real_positive_by_src.items():
        for dst in dsts:
            add_positive(src, dst)

    # C3 SUP real
    for r in sup_real:
        add_positive(r["src"], r["dst"])

    # C3 fixed VAL
    for r in val_fixed:
        add_positive(r["src"], r["dst"])

    # pseudo MP
    for r in mp_pseudo:
        add_positive(r["src"], r["dst"])

    # pseudo SUP
    for r in sup_pseudo:
        add_positive(r["src"], r["dst"])

    # VAL reserved는 positive와 별도로 관리한다.
    reserved_by_src = {}
    for src, dst in val_reserved_pairs:
        reserved_by_src.setdefault(int(src), set()).add(int(dst))

    active = {}
    removed_total = 0

    for src, candidates in unified_pools.items():
        positive_blocked = positive_by_src.get(src, set())
        reserved_blocked = reserved_by_src.get(src, set())
        blocked = positive_blocked | reserved_blocked | {src}
        clean = candidates - blocked
        removed_total += len(candidates) - len(clean)
        active[src] = sorted(clean)

    return active, positive_by_src, reserved_by_src, removed_total


# ============================================================
# MAIN
# ============================================================

def build_round(
    cutoff: int,
    target_start: int,
    target_end: int,
    round_idx: int,
    predecessor_base: bool = False,
):
    print("=" * 100)
    print(
        f"BUILD ROUND DATA | "
        f"{cutoff} -> {target_start}-{target_end} | "
        f"ROUND {round_idx}"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # 1. fixed C3 SUP / VAL
    # --------------------------------------------------------

    (
        c3_obj,
        sup_real,
        val_fixed,
        c3_path,
    ) = load_c3_split(cutoff)

    sup_real = dedup_records(sup_real)
    val_fixed = dedup_records(val_fixed)

    # --------------------------------------------------------
    # Leakage protection: real C3 SUP must be pair-disjoint
    # from fixed C3 VAL. VAL has priority.
    # --------------------------------------------------------
    val_pairs_for_real = pair_set(val_fixed)
    real_sup_before = len(sup_real)
    real_sup_val_overlap = pair_set(sup_real) & val_pairs_for_real

    if real_sup_val_overlap:
        print(
            f"[LEAKAGE FIX] real C3 SUP ∩ fixed VAL = "
            f"{len(real_sup_val_overlap):,} pair(s)"
        )
        print(
            f"[LEAKAGE FIX] examples = "
            f"{sorted(real_sup_val_overlap)[:20]}"
        )
        sup_real = [
            r for r in sup_real
            if (int(r["src"]), int(r["dst"])) not in val_pairs_for_real
        ]

    print(
        f"[LEAKAGE CHECK] real C3 SUP ∩ fixed VAL = "
        f"{len(pair_set(sup_real) & val_pairs_for_real):,}"
    )

    # VAL 평가용 예약 pair: positive가 아니라 NEG sampling 금지 영역
    val_reserved_pairs = {
        (int(src), int(dst))
        for src, dst in c3_obj.get("val_reserved_pairs", set())
    }

    # --------------------------------------------------------
    # 2. unified negative reservoir
    # --------------------------------------------------------

    (
        pool_obj,
        unified_pools,
        real_positive_by_src,
        pool_path,
    ) = load_unified_pool(cutoff)

    # --------------------------------------------------------
    # 3. previous pseudo accumulation
    # --------------------------------------------------------

    current_pseudo = load_previous_pseudo(
        cutoff=cutoff,
        round_idx=round_idx,
    )

    inherited_pseudo, inherited_path, selected_path = load_inherited_pseudo(
        cutoff=cutoff,
        allow_empty=predecessor_base,
    )

    # inherited is a fixed base for every round in this fold;
    # current_pseudo accumulates only pseudo selected inside this fold.
    previous_pseudo = dedup_records(
        list(inherited_pseudo) + list(current_pseudo)
    )

    # --------------------------------------------------------
    # 4. pseudo -> MP / SUP by date
    # --------------------------------------------------------

    inherited_mp, inherited_sup, inherited_outside = split_pseudo_by_time(
        inherited_pseudo,
        cutoff=cutoff,
        target_start=target_start,
        target_end=target_end,
    )

    current_mp, current_sup, current_outside = split_pseudo_by_time(
        current_pseudo,
        cutoff=cutoff,
        target_start=target_start,
        target_end=target_end,
    )

    (
        mp_pseudo,
        sup_pseudo,
        outside_pseudo,
    ) = split_pseudo_by_time(
        previous_pseudo,
        cutoff=cutoff,
        target_start=target_start,
        target_end=target_end,
    )

    # --------------------------------------------------------
    # 5. protect VAL / remove duplicates
    # --------------------------------------------------------

    mp_pseudo, sup_pseudo = clean_pseudo(
        mp_pseudo=mp_pseudo,
        sup_pseudo=sup_pseudo,
        sup_real=sup_real,
        val_fixed=val_fixed,
    )

    # --------------------------------------------------------
    # 6. SUP ALL
    # --------------------------------------------------------

    sup_all = dedup_records(
        list(sup_real) + list(sup_pseudo)
    )

    # --------------------------------------------------------
    # 7. active negative reservoir
    # --------------------------------------------------------

    (
        active_negative_pools,
        positive_by_src,
        reserved_by_src,
        removed_negative_pairs,
    ) = build_active_negative_pool(
        unified_pools=unified_pools,
        real_positive_by_src=real_positive_by_src,
        sup_real=sup_real,
        val_fixed=val_fixed,
        val_reserved_pairs=val_reserved_pairs,
        mp_pseudo=mp_pseudo,
        sup_pseudo=sup_pseudo,
    )

    # --------------------------------------------------------
    # 8. verification
    # --------------------------------------------------------

    val_pairs = pair_set(val_fixed)
    mp_pairs = pair_set(mp_pseudo)
    sup_real_pairs = pair_set(sup_real)
    sup_pseudo_pairs = pair_set(sup_pseudo)
    sup_all_pairs = pair_set(sup_all)

    real_sup_val_leak = sup_real_pairs & val_pairs
    if real_sup_val_leak:
        raise RuntimeError(
            "[LEAKAGE] real C3 SUP ∩ fixed VAL != 0: "
            f"{sorted(real_sup_val_leak)[:20]}"
        )

    assert not (val_pairs & mp_pairs), (
        "VAL ↔ pseudo MP overlap"
    )

    assert not (val_pairs & sup_pseudo_pairs), (
        "VAL ↔ pseudo SUP overlap"
    )

    assert not (mp_pairs & sup_pseudo_pairs), (
        "pseudo MP ↔ pseudo SUP overlap"
    )

    assert not (sup_real_pairs & sup_pseudo_pairs), (
        "real SUP ↔ pseudo SUP overlap"
    )

    # active negative에 positive가 남아 있는지 검사
    leaked = []

    for src, candidates in active_negative_pools.items():
        pos = positive_by_src.get(src, set())
        overlap = set(candidates) & pos

        if overlap:
            leaked.append(
                (src, sorted(overlap)[:10])
            )

    if leaked:
        raise RuntimeError(
            f"Positive leakage in negative pool: "
            f"{leaked[:10]}"
        )

    reserved_leaked = []
    for src, candidates in active_negative_pools.items():
        overlap = set(candidates) & reserved_by_src.get(src, set())
        if overlap:
            reserved_leaked.append((src, sorted(overlap)[:10]))

    if reserved_leaked:
        raise RuntimeError(
            f"VAL reserved leakage in negative pool: "
            f"{reserved_leaked[:10]}"
        )

    # --------------------------------------------------------
    # 9. save
    # --------------------------------------------------------

    out_dir = (
        RUNS_DIR
        / f"cutoff_{cutoff}"
        / f"round_{round_idx:02d}"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "cutoff": cutoff,
        "target": [
            target_start,
            target_end,
        ],
        "round": round_idx,

        # pseudo만 round-specific
        "mp_pseudo": mp_pseudo,
        "sup_pseudo": sup_pseudo,

        # training positive
        "sup_all": sup_all,

        # fixed validation도 검증 편의를 위해 포함
        "val_fixed": val_fixed,
        "val_reserved_pairs": val_reserved_pairs,

        # 현재 round에서 sampling할 NEG reservoir
        "negative_pools": active_negative_pools,

        # provenance
        "sources": {
            "c3_split": str(c3_path),
            "unified_pool": str(pool_path),
            "selected_predecessor": (
                str(selected_path) if selected_path is not None else None
            ),
            "inherited_pseudo": (
                str(inherited_path) if inherited_path is not None else None
            ),
        },

        "counts": {
            "inherited_raw": len(inherited_pseudo),
            "current_pseudo_raw": len(current_pseudo),
            "inherited_mp_raw": len(inherited_mp),
            "inherited_sup_raw": len(inherited_sup),
            "current_mp_raw": len(current_mp),
            "current_sup_raw": len(current_sup),

            "previous_pseudo_raw":
                len(previous_pseudo),

            "mp_pseudo":
                len(mp_pseudo),

            "sup_real":
                len(sup_real),

            "real_sup_removed_due_to_val_overlap":
                real_sup_before - len(sup_real),

            "sup_pseudo":
                len(sup_pseudo),

            "sup_all":
                len(sup_all),

            "val_fixed":
                len(val_fixed),

            "val_reserved_pairs":
                len(val_reserved_pairs),

            "outside_pseudo":
                len(outside_pseudo),

            "negative_src":
                len(active_negative_pools),

            "negative_pairs":
                sum(
                    len(v)
                    for v
                    in active_negative_pools.values()
                ),

            "removed_negative_pairs":
                removed_negative_pairs,
        },
    }

    save_pkl(
        payload,
        out_dir / "round_data.pkl",
    )

    summary = {
        "cutoff": cutoff,
        "target":
            f"{target_start}-{target_end}",
        "round": round_idx,

        **payload["counts"],
    }

    (
        out_dir / "round_data_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 10. print
    # --------------------------------------------------------

    print("\nFIXED")
    print(
        f"C3 SUP real      : "
        f"{len(sup_real):,}"
    )
    print(
        f"C3 VAL fixed     : "
        f"{len(val_fixed):,}"
    )
    print(
        f"VAL reserved     : "
        f"{len(val_reserved_pairs):,}"
    )

    print("\nPSEUDO")
    print(f"inherited raw    : {len(inherited_pseudo):,}")
    print(f"current raw      : {len(current_pseudo):,}")
    print(f"inherited MP raw : {len(inherited_mp):,}")
    print(f"current MP raw   : {len(current_mp):,}")
    print(
        f"previous raw     : "
        f"{len(previous_pseudo):,}"
    )
    print(
        f"MP pseudo        : "
        f"{len(mp_pseudo):,}"
    )
    print(
        f"SUP pseudo       : "
        f"{len(sup_pseudo):,}"
    )
    print(
        f"outside          : "
        f"{len(outside_pseudo):,}"
    )

    print("\nTRAIN")
    print(
        f"SUP ALL          : "
        f"{len(sup_all):,}"
    )
    print(
        f"negative src     : "
        f"{len(active_negative_pools):,}"
    )
    print(
        f"negative pairs   : "
        f"{sum(len(v) for v in active_negative_pools.values()):,}"
    )
    print(
        f"NEG removed      : "
        f"{removed_negative_pairs:,}"
    )

    print("\nVERIFICATION PASSED")
    print(
        "real-SUP/VAL / pseudo-VAL / MP-SUP duplicate / "
        "positive-negative / VAL-reserved-negative leakage checked."
    )

    print(
        "\nsaved:",
        out_dir / "round_data.pkl",
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
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
        "--predecessor-base",
        action="store_true",
        help=(
            "Build a clean target R0 before predecessor selection. "
            "If selected.pkl does not exist, inherited pseudo is empty."
        ),
    )

    args = ap.parse_args()

    build_round(
        cutoff=args.cutoff,
        target_start=args.target_start,
        target_end=args.target_end,
        round_idx=args.round,
        predecessor_base=args.predecessor_base,
    )