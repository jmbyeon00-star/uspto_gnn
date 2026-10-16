from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from data import load_snapshot


TARGET_POOL_SIZE = 1000
SEED = getattr(config, "SEED", 42)

POOL_DIR = ROOT / "fixed_data" / "unified_base_pools"


def relation_maps(snapshot):
    d = snapshot["data"]
    p2ipc, ipc2p, adj = {}, {}, {}

    if ("patent", "has_ipc", "ipc") in d.edge_types:
        for p, i in d["patent", "has_ipc", "ipc"].edge_index.cpu().t().tolist():
            p, i = int(p), int(i)
            p2ipc.setdefault(p, set()).add(i)
            ipc2p.setdefault(i, set()).add(p)

    for et in [("patent", "cites", "patent"), ("patent", "cited_by", "patent")]:
        if et in d.edge_types:
            for s, t in d[et].edge_index.cpu().t().tolist():
                adj.setdefault(int(s), set()).add(int(t))

    return p2ipc, ipc2p, adj


def khop_layers(adj, src, k=3):
    seen = {int(src)}
    frontier = {int(src)}
    layers = []

    for _ in range(k):
        nxt = set()
        for x in frontier:
            nxt |= adj.get(x, set())

        nxt -= seen
        layers.append(nxt)
        seen |= nxt
        frontier = nxt

        if not frontier:
            layers += [set()] * (k - len(layers))
            break

    return layers


def add_until_target(pool, candidates, forbidden, target, rng):
    """중복/positive/self를 제외하고 target까지 필요한 수만 추가한다."""
    candidates = set(candidates) - set(pool) - set(forbidden)
    need = target - len(pool)

    if need <= 0 or not candidates:
        return 0

    if len(candidates) <= need:
        selected = candidates
    else:
        selected = set(map(int, rng.choice(np.asarray(sorted(candidates), dtype=np.int64), size=need, replace=False)))

    pool.update(selected)
    return len(selected)


def expand_fold(cutoff, target=TARGET_POOL_SIZE):
    path = POOL_DIR / f"pool_{cutoff}.pkl"

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "rb") as f:
        obj = pickle.load(f)

    pools = {int(s): set(map(int, pool)) for s, pool in obj["pools"].items()}
    real_by_src = {int(s): set(map(int, ds)) for s, ds in obj.get("real_positive_by_src", {}).items()}

    snapshot = load_snapshot(config.SNAP_DIR, cutoff)
    p2ipc, ipc2p, adj = relation_maps(snapshot)

    rng = np.random.default_rng(SEED + int(cutoff) + int(target))

    before_sizes = np.asarray([len(pool) for pool in pools.values()], dtype=np.int64)

    stats = {
        "cutoff": int(cutoff),
        "target_pool_size": int(target),
        "src_count": len(pools),
        "already_ge_target": 0,
        "expanded_src": 0,
        "citation1_added_total": 0,
        "citation2_added_total": 0,
        "citation3_added_total": 0,
        "ipc_added_total": 0,
        "reached_target_after_expansion": 0,
        "still_below_target": 0,
    }

    for src, pool in pools.items():
        if len(pool) >= target:
            stats["already_ge_target"] += 1
            continue

        original_size = len(pool)
        real = real_by_src.get(src, set())
        forbidden = {src} | real

        layers = khop_layers(adj, src, 3)
        hop1 = layers[0]
        hop2_only = layers[1]
        hop3_only = layers[2]

        same_ipc = set()
        for ipc in p2ipc.get(src, set()):
            same_ipc |= ipc2p.get(ipc, set())

        # 순서 고정:
        # original hard -> citation 1-hop -> new 2-hop -> new 3-hop -> same IPC
        n1 = add_until_target(pool, hop1, forbidden, target, rng)
        n2 = add_until_target(pool, hop2_only, forbidden, target, rng)
        n3 = add_until_target(pool, hop3_only, forbidden, target, rng)
        ni = add_until_target(pool, same_ipc, forbidden, target, rng)

        stats["citation1_added_total"] += n1
        stats["citation2_added_total"] += n2
        stats["citation3_added_total"] += n3
        stats["ipc_added_total"] += ni

        if len(pool) > original_size:
            stats["expanded_src"] += 1

        if len(pool) >= target:
            stats["reached_target_after_expansion"] += 1
        else:
            stats["still_below_target"] += 1

        assert src not in pool
        assert not (pool & real)

    after_sizes = np.asarray([len(pool) for pool in pools.values()], dtype=np.int64)

    # 기존 payload 구조 유지하고 pools만 확장한다.
    obj["pools"] = {int(s): sorted(pool) for s, pool in pools.items()}
    obj.setdefault("policy", {})
    obj["policy"]["fallback_expansion"] = {
        "target_pool_size": int(target),
        "priority": ["citation_1hop", "citation_2hop_only", "citation_3hop_only", "same_ipc"],
        "force_random_fill": False,
        "seed": int(SEED),
    }

    stats.update({
        "before_mean": float(before_sizes.mean()),
        "before_median": float(np.median(before_sizes)),
        "before_lt_target": int((before_sizes < target).sum()),
        "after_mean": float(after_sizes.mean()),
        "after_median": float(np.median(after_sizes)),
        "after_min": int(after_sizes.min()),
        "after_p10": float(np.quantile(after_sizes, 0.10)),
        "after_p25": float(np.quantile(after_sizes, 0.25)),
        "after_p75": float(np.quantile(after_sizes, 0.75)),
        "after_p90": float(np.quantile(after_sizes, 0.90)),
        "after_max": int(after_sizes.max()),
        "after_empty": int((after_sizes == 0).sum()),
        "after_lt_100": int((after_sizes < 100).sum()),
        "after_lt_500": int((after_sizes < 500).sum()),
        "after_lt_1000": int((after_sizes < 1000).sum()),
    })

    # 최초 실행 시 확장 전 원본을 그대로 backup한다.
    backup = POOL_DIR / f"pool_{cutoff}_before_fallback.pkl"
    if not backup.exists():
        backup.write_bytes(path.read_bytes())

    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(path)

    summary_path = POOL_DIR / f"pool_{cutoff}_fallback_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return stats


def verify_fold(cutoff):
    path = POOL_DIR / f"pool_{cutoff}.pkl"

    with open(path, "rb") as f:
        obj = pickle.load(f)

    real_by_src = {int(s): set(map(int, ds)) for s, ds in obj.get("real_positive_by_src", {}).items()}

    for s, pool in obj["pools"].items():
        src = int(s)
        pool = set(map(int, pool))
        real = real_by_src.get(src, set())

        assert src not in pool, f"self-loop found: {src}"
        assert not (pool & real), f"real-positive collision found: src={src}"

    return True


def main(target):
    summaries = []

    for cutoff, ys, ye in config.TRAIN_FOLDS:
        print("\n" + "=" * 90)
        print(f"EXPAND {cutoff} -> {ys}-{ye} | TARGET={target:,}")
        print("=" * 90)

        stats = expand_fold(cutoff, target)
        verify_fold(cutoff)
        summaries.append(stats)

        print(f"src count            : {stats['src_count']:,}")
        print(f"before median        : {stats['before_median']:,.1f}")
        print(f"before < target      : {stats['before_lt_target']:,}")
        print(f"already >= target    : {stats['already_ge_target']:,}")
        print(f"expanded src         : {stats['expanded_src']:,}")
        print(f"reached target       : {stats['reached_target_after_expansion']:,}")
        print(f"still < target       : {stats['still_below_target']:,}")
        print(f"citation 1-hop added : {stats['citation1_added_total']:,}")
        print(f"citation 2-hop added : {stats['citation2_added_total']:,}")
        print(f"citation 3-hop added : {stats['citation3_added_total']:,}")
        print(f"IPC added            : {stats['ipc_added_total']:,}")
        print(f"after median         : {stats['after_median']:,.1f}")
        print(f"after P10            : {stats['after_p10']:,.1f}")
        print(f"after P25            : {stats['after_p25']:,.1f}")
        print(f"after empty          : {stats['after_empty']:,}")
        print(f"after < 500          : {stats['after_lt_500']:,}")
        print(f"after < 1000         : {stats['after_lt_1000']:,}")
        print("verification          : PASSED")

    summary_path = POOL_DIR / "fallback_expansion_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print("\nSaved:", summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=TARGET_POOL_SIZE)
    args = parser.parse_args()
    main(args.target)
