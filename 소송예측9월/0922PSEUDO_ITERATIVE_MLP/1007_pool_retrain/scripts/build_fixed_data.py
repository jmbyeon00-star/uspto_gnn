from pathlib import Path
import pickle

EXP = Path(__file__).resolve().parent.parent
CUTOFFS = [2004, 2006, 2008, 2010]

def load(path):
    with open(path, "rb") as f: return pickle.load(f)

def save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f: pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def to_pair(r):
    if isinstance(r, dict): return int(r["src"]), int(r["dst"])
    return int(r[0]), int(r[1])

def pair_set(rows):
    return {to_pair(r) for r in rows}

def group(pairs):
    out = {}
    for s, d in pairs: out.setdefault(s, set()).add(d)
    return out

def main(cutoff):
    print("\n" + "=" * 60)
    print(f"CUTOFF: {cutoff}")
    print("=" * 60)

    rd_path = EXP / "source_state" / f"cutoff_{cutoff}" / "round_data.pkl"
    pool_path = EXP / "candidate_pools" / f"cutoff_{cutoff}.pkl"
    out = EXP / "fixed_data" / f"cutoff_{cutoff}"

    rd, po = load(rd_path), load(pool_path)
    raw = {int(s): set(map(int, ds)) for s, ds in po["raw_pools_by_src"].items()}

    sup_all = pair_set(rd.get("sup_all", []))
    sup_real = pair_set(rd.get("sup_real", []))
    mp = pair_set(rd.get("mp_pseudo", []))
    sup_pseudo = pair_set(rd.get("sup_pseudo", []))
    val = pair_set(rd.get("val_fixed", []))
    reserved = pair_set(rd.get("val_reserved_pairs", []))
    hist = pair_set(rd.get("all_positive_pairs", []))

    # round_data의 기존 negative pool은 사용하지 않는다.
    # 새 raw candidate pool에서 positive/leakage를 다시 제거한다.
    known = hist | sup_real | sup_all | mp | sup_pseudo | val
    pos_by_src, res_by_src, val_by_src = group(known), group(reserved), group(val)

    train_srcs = sorted({s for s, _ in sup_all})
    val_srcs = sorted(val_by_src)

    train = {}
    for s in train_srcs:
        blocked = {s} | pos_by_src.get(s, set()) | res_by_src.get(s, set())
        train[s] = sorted(raw.get(s, set()) - blocked)

    val_pool = {}
    for s in val_srcs:
        # 모든 VAL GT는 negative에서 제거하고 GT는 학습 코드에서 별도 score.
        blocked = {s} | pos_by_src.get(s, set()) | res_by_src.get(s, set())
        val_pool[s] = sorted(raw.get(s, set()) - blocked)

    # Retrieval recall은 GT 제거 전 raw pool 기준.
    val_retrieved = {(s, d): d in raw.get(s, set()) for s, d in val}

    # Safety audit
    train_leak = sum(len(set(ds) & ({s} | pos_by_src.get(s, set()) |
                                    res_by_src.get(s, set()))) for s, ds in train.items())
    val_leak = sum(len(set(ds) & ({s} | pos_by_src.get(s, set()) |
                                  res_by_src.get(s, set()))) for s, ds in val_pool.items())
    small = {s: len(ds) for s, ds in train.items() if len(ds) < 5}

    if train_leak: raise RuntimeError(f"TRAIN leakage: {train_leak}")
    if val_leak: raise RuntimeError(f"VAL leakage: {val_leak}")
    if small: raise RuntimeError(f"Hard5 불가능 TRAIN SRC: {small}")

    payload = {
        "cutoff": cutoff,
        "source_round_data": str(rd_path),
        "raw_candidate_pool": str(pool_path),
        "sup_all": rd.get("sup_all", []),
        "val_fixed": rd.get("val_fixed", []),
        "mp_pseudo": rd.get("mp_pseudo", []),
        "sup_pseudo": rd.get("sup_pseudo", []),
        "val_reserved_pairs": rd.get("val_reserved_pairs", []),
        "train_negative_pools": train,
        "val_candidate_pools": val_pool,
        "val_retrieved": val_retrieved,
        "policy": {
            "train": "candidate-aware pool; sample 5 negatives per positive; no global random",
            "val": "score GT separately against full filtered candidate pool"
        }
    }

    save(train, out / "train_negative_pools.pkl")
    save(val_pool, out / "val_candidate_pools.pkl")
    save(payload, out / "fixed_data.pkl")

    tr_sizes = [len(x) for x in train.values()]
    va_sizes = [len(x) for x in val_pool.values()]
    hit, n = sum(val_retrieved.values()), len(val_retrieved)

    print(f"SUP_ALL          : {len(sup_all):,}")
    print(f"VAL              : {n:,}")
    print(f"VAL retrieved    : {hit:,}/{n:,} ({100*hit/n:.2f}%)" if n else "VAL retrieved    : 0/0")
    print(f"TRAIN src        : {len(train):,}")
    print(f"VAL src          : {len(val_pool):,}")
    print(f"TRAIN pool min/max: {min(tr_sizes):,} / {max(tr_sizes):,}")
    print(f"VAL pool min/max  : {min(va_sizes):,} / {max(va_sizes):,}")
    print(f"TRAIN leakage    : {train_leak}")
    print(f"VAL leakage      : {val_leak}")
    print(f"saved            : {out}")

if __name__ == "__main__":
    for cutoff in CUTOFFS: main(cutoff)
    print("\nALL DONE")
