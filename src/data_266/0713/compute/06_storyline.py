"""compute/06_storyline.py → results/storyline.json
Auto-generate narrative storyline from all other JSON results.
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import load_context, save_json, RESULTS_DIR


def _load(name):
    p = RESULTS_DIR / name
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def build_storyline(ctx: dict) -> None:
    print("[06_storyline] Generating narrative...")

    pred  = _load("prediction.json")
    expln = _load("explanation.json")
    cf    = _load("counterfactual.json")
    ep    = _load("evidence_path.json")

    p      = pred.get("probability", 0.0)
    src    = pred.get("query_patent", "?")
    dst    = pred.get("target_patent", "?")
    plt_co = pred.get("plaintiff_company", "Unknown")
    def_co = pred.get("defendant_company", "Unknown")
    conf   = pred.get("confidence", "unknown")
    result = pred.get("prediction", "no_sue")

    # ── Main reason ───────────────────────────────────────────────────────────
    gnn_imp  = expln.get("gnn_explainer", {}).get("edge_importance", {})
    top_edges_gnn = sorted(gnn_imp, key=lambda k: -gnn_imp.get(k, 0))[:3]
    n_top = len(top_edges_gnn)

    by_type = cf.get("by_type", {})
    worst_type = min(by_type, key=lambda k: by_type[k].get("delta", 0)) if by_type else "citation"
    worst_delta = by_type.get(worst_type, {}).get("delta", 0.0)
    worst_after = by_type.get(worst_type, {}).get("after", p)

    path_len = ep.get("hops", 0)
    path_score = ep.get("score", 0.0)

    ipc_src  = ctx["src_ipc"][:2]
    ipc_dst  = ctx["dst_ipc"][:2]
    ipc_shared = list(set(ipc_src) & set(ipc_dst))

    main_reason = (
        f"GNNExplainer identified {n_top} critical citation edges "
        f"between patents, suggesting strong citation similarity."
    )
    if n_top == 0:
        main_reason = "The model found limited direct citation links; ownership overlap is the primary signal."

    secondary_reason = ""
    if ipc_shared:
        secondary_reason = (
            f"Both patents share IPC class {ipc_shared[0]}, "
            f"indicating overlapping technology domains."
        )
    elif ipc_src and ipc_dst:
        secondary_reason = (
            f"Plaintiff's IPC: {', '.join(ipc_src)}, Defendant's IPC: {', '.join(ipc_dst)}. "
            f"No direct overlap, but the GNN captured indirect similarity."
        )

    cf_text = (
        f"Removing all {worst_type} edges drops P(sue) from "
        f"{p:.3f} to {worst_after:.3f} (Δ={worst_delta:+.3f}), "
        f"confirming {worst_type} as the most causal factor."
    )

    conclusion = (
        f"{'⚖️ Litigation predicted' if result == 'sue' else '✅ No litigation predicted'} "
        f"({p:.1%} confidence={conf}). "
        f"The strongest evidence is the {worst_type} relationship "
        f"between {src} ({plt_co}) and {dst} ({def_co}). "
        f"Evidence path length: {path_len} hops (score={path_score:.3f})."
    )

    # ── Cards ─────────────────────────────────────────────────────────────────
    cards = [
        {
            "step": 1, "title": "Prediction",
            "icon": "⚖️",
            "value": f"{p:.1%}",
            "text": (
                f"P(sue) = **{p:.4f}** — "
                f"{'above' if p >= pred.get('threshold', 0.5) else 'below'} decision threshold "
                f"({pred.get('threshold', 0.5):.2f}). "
                f"Confidence: {conf}."
            ),
        },
        {
            "step": 2, "title": "Evidence",
            "icon": "🔍",
            "value": f"{n_top} key edges",
            "text": main_reason + (" " + secondary_reason if secondary_reason else ""),
        },
        {
            "step": 3, "title": "Counterfactual",
            "icon": "🔄",
            "value": f"Δ={worst_delta:+.3f}",
            "text": cf_text,
        },
        {
            "step": 4, "title": "Conclusion",
            "icon": "✅",
            "value": None,
            "text": conclusion,
        },
    ]

    result_json = {
        "summary":              f"Patent {src} is predicted to sue {dst} with {p:.1%} probability.",
        "main_reason":          main_reason,
        "secondary_reason":     secondary_reason,
        "counterfactual_result": cf_text,
        "final_conclusion":     conclusion,
        "cards":                cards,
        "created_at":           ctx["created_at"],
    }
    save_json(result_json, "storyline.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--khop", type=int, default=2)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    ctx = load_context(args.pair, args.khop, args.topk)
    build_storyline(ctx)
