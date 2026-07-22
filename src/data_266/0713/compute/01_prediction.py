"""compute/01_prediction.py → results/prediction.json

Full ranking approach:
  - query = src patent (test plaintiff)
  - rank dst patent (ground truth defendant) among ALL N patents
  - top_k = top-K from full ranking (includes ground truth)
  - XAI is done on the (src, dst) ground truth pair
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import load_context, save_json, sigmoid


def build_prediction(ctx: dict) -> None:
    print("[01_prediction] Full ranking...")
    h_emb    = ctx["h_emb"]
    src_idx  = ctx["src_idx"]
    dst_idx  = ctx["dst_idx"]
    topk     = ctx["topk"]
    pat_list = ctx["pat_list"]
    p_base   = ctx["p_base"]
    state    = ctx["state"]
    df_asgn  = ctx["df_asgn"]

    N = len(pat_list)

    def get_owner(pat_label):
        if df_asgn is None:
            return "Unknown"
        try:
            col  = df_asgn.columns[0]
            rows = df_asgn[df_asgn[col] == pat_label]
            return str(rows.iloc[0, 1]) if len(rows) else "Unknown"
        except Exception:
            return "Unknown"

    # ── Full ranking: score ALL patents against src ───────────────────────────
    logits = h_emb @ h_emb[src_idx]          # (N,)
    logits[src_idx] = -np.inf                 # exclude self
    probs  = 1 / (1 + np.exp(-logits))        # sigmoid → (N,)

    # Rank of ground truth defendant (1-indexed)
    # rank = number of patents scoring HIGHER than dst + 1
    dst_score = float(probs[dst_idx])
    gt_rank   = int((probs > dst_score).sum()) + 1   # 1-indexed

    # Top-K from full ranking
    top_indices = np.argsort(-probs)          # descending
    top_k = []
    for idx in top_indices:
        i = int(idx)
        if i == src_idx:
            continue
        top_k.append({
            "rank":         len(top_k) + 1,
            "patent_id":    f"pat_{i}",
            "patent_label": pat_list[i],
            "company":      get_owner(pat_list[i]),
            "probability":  round(float(probs[i]), 4),
            "is_ground_truth": i == dst_idx,
        })
        if len(top_k) >= topk:
            break

    # Check if ground truth appears in top_k window; if not, append it
    gt_in_topk = any(t["is_ground_truth"] for t in top_k)
    if not gt_in_topk:
        top_k.append({
            "rank":         gt_rank,
            "patent_id":    f"pat_{dst_idx}",
            "patent_label": pat_list[dst_idx],
            "company":      get_owner(pat_list[dst_idx]),
            "probability":  round(dst_score, 4),
            "is_ground_truth": True,
        })

    # Confidence: based on how high the ground truth ranks
    if gt_rank <= 5:
        confidence = "high"
    elif gt_rank <= 20:
        confidence = "medium"
    else:
        confidence = "low"

    best_thr = float(state.get("best_threshold", 0.5))

    print(f"    Ground truth rank: {gt_rank} / {N}   P={dst_score:.4f}")
    print(f"    Top-1: {pat_list[top_indices[0]]}  P={probs[top_indices[0]]:.4f}")

    result = {
        # ── Query & Ground Truth ──────────────────────────────────────────────
        "query_patent":        ctx["src_label"],
        "query_patent_id":     f"pat_{src_idx}",
        "target_patent":       ctx["dst_label"],      # ground truth
        "target_patent_id":    f"pat_{dst_idx}",
        "plaintiff_company":   ctx["src_owners"][0] if ctx["src_owners"] else "Unknown",
        "defendant_company":   ctx["dst_owners"][0] if ctx["dst_owners"] else "Unknown",

        # ── Scores ───────────────────────────────────────────────────────────
        "probability":         round(dst_score, 6),   # ground truth pair score
        "threshold":           round(best_thr, 4),
        "prediction":          "sue" if dst_score >= best_thr else "no_sue",
        "ground_truth":        "positive",            # test set = real lawsuits
        "correct":             dst_score >= best_thr,

        # ── Ranking ──────────────────────────────────────────────────────────
        "rank":                gt_rank,
        "total_patents":       N,
        "rank_percentile":     round(100 * (1 - gt_rank / N), 1),  # top X%
        "confidence":          confidence,

        # ── Top-K ────────────────────────────────────────────────────────────
        "top_k":               top_k,

        "main_evidence":       "citation",   # updated by 03_explanation
        "created_at":          ctx["created_at"],
    }
    save_json(result, "prediction.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--khop", type=int, default=2)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    ctx = load_context(args.pair, args.khop, args.topk)
    build_prediction(ctx)
