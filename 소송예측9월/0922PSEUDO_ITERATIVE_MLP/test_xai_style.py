"""
XAI-style 1000-way evaluation on snap_2010.

XAI의 s5_test_neg_final_mixed.npy neg set을 그대로 사용.
8월 2010 snapshot 인덱스 → patent ID → 9월 2010 snapshot 인덱스로 변환.

GT: s5_test_src/dst.npy 중 9월 snap_2010에 존재하는 쌍 (~389/404)
Neg: s5_test_neg_final_mixed.npy 변환, 9월에 없는 것은 random으로 대체

Metrics: Hits@10, Hits@20, Hits@50, MRR, AUROC — XAI evaluation.py와 동일
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from data import EDGE_TYPES, StaticFeatureStore, load_snapshot, build_features, canonical_sue
from model import Model


# ============================================================
# 경로
# ============================================================

SNAP8_DIR = Path(
    "/home2/jmbyeon/developments/workspace/apps/uspto_litigation/"
    "소송예측모델 개발/소송예측8월/snapshots"
)
PAT_IDS_8W_PATH = SNAP8_DIR / "2010snapshot/patents.npy"
TEST_SRC_PATH   = SNAP8_DIR / "s5_test_src.npy"
TEST_DST_PATH   = SNAP8_DIR / "s5_test_dst.npy"
TEST_NEG_PATH   = SNAP8_DIR / "s5_test_neg_final_mixed.npy"

SNAP_YEAR = 2010

CHECKPOINTS = {
    "cutoff_2004_R09": ROOT / "runs" / "cutoff_2004" / "round_09" / "checkpoint.pt",
    "cutoff_2006_R00": ROOT / "runs" / "cutoff_2006" / "round_00" / "checkpoint.pt",
    "cutoff_2008_R06": ROOT / "runs" / "cutoff_2008" / "round_06" / "checkpoint.pt",
}

HITS_K = [1, 10, 20, 50, 100]
SEED   = config.SEED

OUT_DIR = ROOT / "runs" / "xai_style_test_2010"


# ============================================================
# Helpers
# ============================================================

def make_model(feats):
    dims = {k: int(v.shape[1]) for k, v in feats.items()}
    return Model(dims, config.HIDDEN, EDGE_TYPES, config.DROPOUT)


def build_snap10_edge_dict(snapshot):
    """snap_2010의 모든 sue 엣지(year <= 2010)를 MP에 포함. XAI와 동일."""
    data = snapshot["data"]
    out = {}

    for etype in EDGE_TYPES:
        if etype in {("patent", "sue", "patent"), ("patent", "sued_by", "patent")}:
            continue
        if etype not in data.edge_types:
            continue
        out[etype] = data[etype].edge_index.cpu()

    ei, _, _ = canonical_sue(snapshot)
    out[("patent", "sue",      "patent")] = ei
    out[("patent", "sued_by",  "patent")] = torch.flip(ei, dims=[0])
    return out


@torch.no_grad()
def rank_xai(model, z, src, true_dst, neg_indices, device):
    """
    XAI evaluation.py와 동일한 rank 계산.
    rank = (scores >= true_score).sum()  (1-based, ties → 동등 처리)
    """
    all_dst = [true_dst] + [int(x) for x in neg_indices]
    src_t = torch.full((len(all_dst),), src, dtype=torch.long, device=device)
    dst_t = torch.tensor(all_dst, dtype=torch.long, device=device)
    scores = model.score(z, src_t, dst_t).cpu().numpy().astype(np.float32)

    true_score = float(scores[0])
    rank       = int((scores >= scores[0]).sum())  # XAI 동일 공식
    return rank, true_score, scores


# ============================================================
# 8월→9월 인덱스 변환
# ============================================================

def build_index_map(pat_ids_8w: np.ndarray, pat2idx_9w: dict):
    """
    8월 2010 snapshot 인덱스 → 9월 2010 snapshot 인덱스.
    9월에 없으면 -1.
    """
    idx_8w_to_9w = np.full(len(pat_ids_8w), -1, dtype=np.int64)
    for i, pid in enumerate(pat_ids_8w):
        v = pat2idx_9w.get(str(pid))
        if v is not None:
            idx_8w_to_9w[i] = v
    return idx_8w_to_9w


def convert_neg_row(neg_row_8w, idx_map, n_pat_9w, rng, blocked):
    """
    neg_row_8w: shape (1000,) — 8월 인덱스
    9월에 없는 항목은 random으로 대체 (blocked 제외).
    """
    converted = []
    missing   = 0
    for idx8 in neg_row_8w:
        idx9 = int(idx_map[int(idx8)])
        if idx9 >= 0 and idx9 not in blocked:
            converted.append(idx9)
        else:
            missing += 1

    # missing만큼 random으로 채움
    fill_blocked = blocked | set(converted)
    while len(converted) < len(neg_row_8w):
        x = int(rng.integers(0, n_pat_9w))
        if x not in fill_blocked:
            converted.append(x)
            fill_blocked.add(x)

    return converted[:len(neg_row_8w)], missing


# ============================================================
# 평가 루프
# ============================================================

def evaluate(ckpt_name, ckpt_path, snap_9w, feats, ed, idx_map,
             test_pairs_9w, neg_9w_all, device, rng_seed):

    print(f"\n{'='*70}")
    print(f"Evaluating: {ckpt_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*70}")

    counts = {k: int(v.shape[0]) for k, v in feats.items()}
    fd = {k: v.to(device) for k, v in feats.items()}
    gd = {"edge_dict": {k: v.to(device) for k, v in ed.items()}, "counts": counts}

    model = make_model(feats).to(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        z = model.encode(fd, gd)
    print(f"z shape: {tuple(z.shape)}")

    rng = np.random.default_rng(rng_seed)

    ranks      = []
    all_labels = []
    all_scores = []

    for i, (src, dst, src_pid, dst_pid, row_idx) in enumerate(test_pairs_9w, 1):
        blocked = {src, dst}
        neg_row, n_missing = convert_neg_row(
            neg_9w_all[row_idx], idx_map,
            int(z.shape[0]), rng, blocked
        )

        rank, true_score, scores = rank_xai(model, z, src, dst, neg_row, device)

        ranks.append(rank)
        all_labels.append(1)
        all_scores.append(float(scores[0]))
        for ns in scores[1:]:
            all_labels.append(0)
            all_scores.append(float(ns))

        print(
            f"  [{i:03d}/{len(test_pairs_9w)}] {src_pid} -> {dst_pid} | "
            f"rank={rank}  missing_neg={n_missing}"
        )

    ranks_np = np.array(ranks, dtype=np.int64)
    auroc    = float(roc_auc_score(all_labels, all_scores))

    metrics = {
        "checkpoint":  ckpt_name,
        "source":      str(ckpt_path),
        "snap_year":   SNAP_YEAR,
        "test_pairs":  len(ranks),
        "MRR":         float(np.mean(1.0 / ranks_np)),
        "MeanRank":    float(np.mean(ranks_np)),
        "MedianRank":  float(np.median(ranks_np)),
        "AUROC":       auroc,
    }
    for k in HITS_K:
        metrics[f"Hits@{k}"] = float(np.mean(ranks_np <= k))

    return metrics


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cutoff", choices=["2004", "2006", "2008", "all"], default="all",
        help="평가할 cutoff (기본: all)",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # --------------------------------------------------------
    # 1. 8월 테스트 파일 로드 (인덱스 변환 전)
    # --------------------------------------------------------
    print("\n[1] 8월 XAI test 파일 로드")
    pat_ids_8w = np.load(PAT_IDS_8W_PATH, allow_pickle=True).astype(str)
    src_8w     = np.load(TEST_SRC_PATH).astype(np.int64)
    dst_8w     = np.load(TEST_DST_PATH).astype(np.int64)
    neg_8w     = np.load(TEST_NEG_PATH).astype(np.int64)   # (404, 1000)
    print(f"  GT pairs: {len(src_8w)}  |  neg shape: {neg_8w.shape}")

    # --------------------------------------------------------
    # 2. 9월 snap_2010 로드
    # --------------------------------------------------------
    print(f"\n[2] 9월 snap_{SNAP_YEAR} 로드")
    snap_9w   = load_snapshot(config.SNAP_DIR, SNAP_YEAR)
    pat2idx_9w = {str(k): int(v) for k, v in snap_9w["pat2idx"].items()}
    n_pat_9w   = len(pat2idx_9w)
    print(f"  patent nodes: {n_pat_9w:,}")

    # --------------------------------------------------------
    # 3. 8월→9월 인덱스 맵
    # --------------------------------------------------------
    print("\n[3] 인덱스 변환 맵 빌드")
    idx_map = build_index_map(pat_ids_8w, pat2idx_9w)
    mapped  = int((idx_map >= 0).sum())
    print(f"  8월 전체: {len(idx_map):,}  |  9월에 존재: {mapped:,}")

    # --------------------------------------------------------
    # 4. GT pair 변환 (9월 인덱스로)
    # --------------------------------------------------------
    print("\n[4] GT pair 변환")
    test_pairs_9w = []
    skipped = 0
    for row_i, (s8, d8) in enumerate(zip(src_8w, dst_8w)):
        s9 = int(idx_map[int(s8)])
        d9 = int(idx_map[int(d8)])
        if s9 < 0 or d9 < 0:
            skipped += 1
            continue
        src_pid = pat_ids_8w[int(s8)]
        dst_pid = pat_ids_8w[int(d8)]
        test_pairs_9w.append((s9, d9, src_pid, dst_pid, row_i))

    print(f"  변환 성공: {len(test_pairs_9w)} / {len(src_8w)}  (스킵: {skipped})")

    # --------------------------------------------------------
    # 5. Features + edge dict (snap_2010 — 모든 sue 포함)
    # --------------------------------------------------------
    print("\n[5] Features + edge dict 빌드")
    static = StaticFeatureStore.load(config.DATA_GNN)
    feats, info = build_features(snap_9w, static)
    ed = build_snap10_edge_dict(snap_9w)

    sue_edges = ed[("patent", "sue", "patent")]
    print(f"  sue MP edges: {sue_edges.shape[1]:,}")
    print(f"  patent feature coverage: {info['patent_feature_coverage']:.4f}")

    # --------------------------------------------------------
    # 6. 각 checkpoint 평가
    # --------------------------------------------------------
    if args.cutoff == "all":
        targets = list(CHECKPOINTS.items())
    else:
        targets = [(k, v) for k, v in CHECKPOINTS.items() if args.cutoff in k]

    all_results = []
    for ckpt_name, ckpt_path in targets:
        if not ckpt_path.exists():
            print(f"\n[SKIP] {ckpt_name}: {ckpt_path} 없음")
            continue

        metrics = evaluate(
            ckpt_name, ckpt_path,
            snap_9w, feats, ed, idx_map,
            test_pairs_9w, neg_8w, device,
            rng_seed=SEED,
        )
        all_results.append(metrics)

        out_json = OUT_DIR / f"metrics_{ckpt_name}.json"
        out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\n  saved: {out_json}")

    # --------------------------------------------------------
    # 7. 요약 출력
    # --------------------------------------------------------
    print()
    print("=" * 90)
    print("SUMMARY  (XAI-style 1000-way | snap_2010 | s5_test_neg_final_mixed)")
    print("=" * 90)

    # XAI 기준값 (sw_C4_fold02_T2006, 8월 2010 snapshot)
    xai_ref = {
        "checkpoint": "XAI sw_C4_fold02_T2006 (8월)",
        "test_pairs": 404,
        "MRR": 0.1470, "AUROC": None,
        "Hits@10": 0.3317, "Hits@20": None, "Hits@50": 0.6955, "Hits@100": None,
    }

    header = f"{'Model':<28} {'pairs':>5} {'MRR':>8} {'H@10':>8} {'H@20':>8} {'H@50':>8} {'H@100':>8} {'AUROC':>8}"
    print(header)
    print("-" * len(header))

    def fmt_row(r):
        return (
            f"{r['checkpoint']:<28}"
            f" {r.get('test_pairs', '-'):>5}"
            f" {r.get('MRR', float('nan')):>8.4f}"
            f" {r.get('Hits@10', float('nan')):>8.4f}"
            f" {r.get('Hits@20', r.get('Hits@20', float('nan'))):>8.4f}"
            f" {r.get('Hits@50', float('nan')):>8.4f}"
            f" {r.get('Hits@100', float('nan')):>8.4f}"
            f" {r.get('AUROC', float('nan')):>8.4f}"
        )

    ref_row = (
        f"{'XAI sw_C4_fold02_T2006':<28}"
        f" {'404':>5}"
        f" {'0.1470':>8}"
        f" {'0.3317':>8}"
        f" {'N/A':>8}"
        f" {'0.6955':>8}"
        f" {'N/A':>8}"
        f" {'N/A':>8}"
    )
    print(ref_row)
    print("-" * len(header))
    for r in all_results:
        print(fmt_row(r))
    print("=" * 90)

    # 전체 결과 저장
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps({"xai_ref": xai_ref, "results": all_results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nsummary: {summary_path}")


if __name__ == "__main__":
    main()
