from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ------------------------------------------------------------
# Path / imports
# ------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

# iterative_distmult를 최우선 import 경로로 고정
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE))

import config
from data import EDGE_TYPES, StaticFeatureStore, load_snapshot, build_features
from model import Model
import model as model_module

print("TEST MODEL FILE:", model_module.__file__)



CUTOFF = 2012

CHECKPOINT = PROJECT / "runs_distmult_fixed/cutoff_2010/round_02/checkpoint.pt"
INHERITED_PATH = PROJECT / "runs_distmult_fixed/final_test_2012/inherited_from_2010_R2.pkl"
TEST_GT_PATH = PROJECT / "fixed_data/test/test_gt_2013_2014.pkl"
OUT_DIR = PROJECT / "runs_distmult_fixed/final_test_2012"
OUT_DIR.mkdir(parents=True, exist_ok=True)


CHUNK_SIZE = 100_000
HITS_K = [1, 10, 50, 100, 500, 1000, 10000]
SUE = ("patent", "sue", "patent")


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def make_edge_dict(snapshot):
    data = snapshot["data"]
    return {e: data[e].edge_index.cpu() for e in EDGE_TYPES if e in data.edge_types}


def make_model(feats):
    dims = {k: int(v.shape[1]) for k, v in feats.items()}
    return Model(dims, config.HIDDEN, EDGE_TYPES, config.DROPOUT)


def add_mp(edge_dict, pairs):
    old = edge_dict.get(SUE, torch.empty((2, 0), dtype=torch.long)).cpu()
    if old.numel():
        old = torch.unique(old.t(), dim=0).t().contiguous()

    pairs = sorted({(int(s), int(d)) for s, d in pairs})
    add = (
        torch.tensor(pairs, dtype=torch.long).t().contiguous()
        if pairs else torch.empty((2, 0), dtype=torch.long)
    )

    merged = torch.cat([old, add], dim=1)
    if merged.numel():
        merged = torch.unique(merged.t(), dim=0).t().contiguous()

    edge_dict[SUE] = merged
    return {
        "historical": int(old.shape[1]),
        "pseudo_requested": len(pairs),
        "actually_new": int(merged.shape[1] - old.shape[1]),
        "final": int(merged.shape[1]),
    }


@torch.no_grad()
def full_rank(model, z, src, true_dst, n_pat, blocked_dst, device):
    src, true_dst = int(src), int(true_dst)
    ts = torch.tensor([src], dtype=torch.long, device=device)
    td = torch.tensor([true_dst], dtype=torch.long, device=device)
    true_score = model.score(z, ts, td).item()

    blocked = {int(x) for x in blocked_dst}
    blocked.add(src)
    blocked.discard(true_dst)

    higher = 0
    valid_candidates = 0

    for start in range(0, n_pat, CHUNK_SIZE):
        cand = np.arange(start, min(start + CHUNK_SIZE, n_pat), dtype=np.int64)
        if blocked:
            cand = cand[np.array([int(x) not in blocked for x in cand], dtype=bool)]
        if not len(cand):
            continue

        dst = torch.as_tensor(cand, dtype=torch.long, device=device)
        src_t = torch.full((len(cand),), src, dtype=torch.long, device=device)
        scores = model.score(z, src_t, dst)

        higher += int((scores > true_score).sum().item())
        valid_candidates += len(cand)

    return 1 + higher, valid_candidates, true_score


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 1) 2012 snapshot + features
    snap = load_snapshot(config.SNAP_DIR, CUTOFF)
    pat2idx = {str(k): int(v) for k, v in snap["pat2idx"].items()}
    n_pat = len(pat2idx)

    static = StaticFeatureStore.load(config.DATA_GNN)
    feats, _ = build_features(snap, static)
    edge_dict = make_edge_dict(snap)

    # 2) Add pseudo inherited from 2010 R2
    inherited = load_pkl(INHERITED_PATH)
    mp_stats = add_mp(edge_dict, inherited["inherited_pairs"])

    print(f"patent nodes      : {n_pat:,}")
    print(f"historical GT MP  : {mp_stats['historical']:,}")
    print(f"inherited pseudo  : {mp_stats['pseudo_requested']:,}")
    print(f"actually new      : {mp_stats['actually_new']:,}")
    print(f"final MP          : {mp_stats['final']:,}")

    # 3) Graph tensors
    fd = {k: v.to(device) for k, v in feats.items()}
    gd = {
        "edge_dict": {k: v.to(device) for k, v in edge_dict.items()},
        "counts": {k: int(v.shape[0]) for k, v in feats.items()},
    }

    # 4) Load 2010 R2 and encode 2012 graph
    model = make_model(feats).to(device)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        z = model.encode(fd, gd)

    print("checkpoint        :", CHECKPOINT)
    print("patent embedding  :", tuple(z.shape))

    # 5) 2013-2014 test GT -> 2012 node indices
    test_obj = load_pkl(TEST_GT_PATH)
    patent_pairs = sorted({(str(a), str(b)) for a, b in test_obj["test_patent_pairs"]})

    test_pairs, missing = [], []
    for a, b in patent_pairs:
        if a not in pat2idx or b not in pat2idx:
            missing.append((a, b))
        else:
            test_pairs.append((pat2idx[a], pat2idx[b], a, b))

    if missing:
        raise RuntimeError(
            f"Test GT 중 2012 snapshot에 없는 pair {len(missing)}개 발견: {missing[:10]}"
        )

    print("Test GT pairs     :", len(test_pairs))

    # 6) Filter known positives:
    #    historical GT MP + inherited pseudo + other test positives
    known_by_src = {}
    for s, d in edge_dict[SUE].t().tolist():
        known_by_src.setdefault(int(s), set()).add(int(d))
    for s, d, _, _ in test_pairs:
        known_by_src.setdefault(int(s), set()).add(int(d))

    # 7) Filtered full ranking
    rows = []
    print("\n" + "=" * 80)
    print("FILTERED FULL RANKING")
    print("=" * 80)

    for i, (src, dst, plt_pat, def_pat) in enumerate(test_pairs, 1):
        rank, n_candidates, score = full_rank(
            model, z, src, dst, n_pat, known_by_src.get(src, set()), device
        )

        row = {
            "plt_patent": plt_pat,
            "def_patent": def_pat,
            "src": src,
            "dst": dst,
            "rank": rank,
            "n_candidates": n_candidates,
            "true_score": score,
            "RR": 1.0 / rank,
        }
        for k in HITS_K:
            row[f"Hits@{k}"] = int(rank <= k)
        rows.append(row)

        print(
            f"[{i:03d}/{len(test_pairs)}] {plt_pat} -> {def_pat} | "
            f"rank={rank:,}/{n_candidates:,}"
        )

    # 8) Metrics + save
    df = pd.DataFrame(rows)
    ranks = df["rank"].to_numpy(dtype=np.int64)

    metrics = {
        "cutoff": 2012,
        "test_period": [2013, 2014],
        "source_model": "2010 R2",
        "ranking": "filtered_full",
        "test_pairs": int(len(df)),
        "patent_nodes": int(n_pat),
        "historical_gt_mp": mp_stats["historical"],
        "inherited_pseudo_mp": mp_stats["pseudo_requested"],
        "final_mp": mp_stats["final"],
        "MRR": float(np.mean(1.0 / ranks)),
        "MeanRank": float(np.mean(ranks)),
        "MedianRank": float(np.median(ranks)),
    }
    for k in HITS_K:
        metrics[f"Hits@{k}"] = float(np.mean(ranks <= k))

    csv_path = OUT_DIR / "test_pair_ranks.csv"
    json_path = OUT_DIR / "final_test_metrics.json"
    df.sort_values("rank").to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("FINAL TEST RESULT")
    print("=" * 80)
    print(f"Test pairs  : {len(df)}")
    print(f"MRR         : {metrics['MRR']:.8f}")
    print(f"Mean Rank   : {metrics['MeanRank']:,.2f}")
    print(f"Median Rank : {metrics['MedianRank']:,.2f}")
    for k in HITS_K:
        print(f"Hits@{k:<5} : {metrics[f'Hits@{k}']:.6f}")
    print("\nrank CSV :", csv_path)
    print("metrics  :", json_path)


if __name__ == "__main__":
    main()
