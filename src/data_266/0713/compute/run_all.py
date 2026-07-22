"""compute/run_all.py — Run the full XAI pipeline in order.
Usage:
    cd /home2/.../data_266/0713/
    python compute/run_all.py --pair 0 --khop 2 --topk 10
"""
import sys, argparse, importlib.util
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from common import load_context, save_json, RESULTS_DIR

BASE = Path(__file__).parent


def _import(fname: str):
    spec = importlib.util.spec_from_file_location(fname, BASE / fname)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_metadata(ctx: dict) -> None:
    n_pat = len(ctx["pat_list"])
    n_co  = len(ctx["co_list"])
    n_ipc = len(ctx["ipc_list"])
    save_json({
        "model":       "SimpleCatHeteroGAT",
        "dataset":     "USPTO Patent Litigation",
        "hidden_dim":  128,
        "output_dim":  64,
        "heads":       [8, 4, 4, 2],
        "layers":      4,
        "node_types":  ["patent", "company", "ipc"],
        "edge_types":  8,
        "n_patents":   n_pat,
        "n_companies": n_co,
        "n_ipc":       n_ipc,
        "pair":        ctx["pair"],
        "khop":        ctx["khop"],
        "topk":        ctx["topk"],
        "created_at":  ctx["created_at"],
    }, "metadata.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--khop", type=int, default=2)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Step numbers to skip, e.g. --skip 3 4")
    args = ap.parse_args()
    skip = {int(s) for s in (args.skip or [])}

    print(f"\n{'='*60}")
    print(f"Patent Litigation XAI Pipeline")
    print(f"pair={args.pair}  khop={args.khop}  topk={args.topk}")
    print(f"{'='*60}\n")

    ctx = load_context(args.pair, args.khop, args.topk)

    steps = [
        (0,  "Graph",          "00_graph.py",          "build_graph"),
        (10, "Full Graph",     "00b_full_graph.py",    "build_full_graph"),
        (1,  "Prediction",     "01_prediction.py",     "build_prediction"),
        (2,  "Attention",      "02_attention.py",      "build_attention"),
        (3,  "Explanation",    "03_explanation.py",    "build_explanation"),
        (4,  "Counterfactual", "04_counterfactual.py", "build_counterfactual"),
        (5,  "Evidence Path",  "05_evidence_path.py",  "build_evidence_path"),
        (6,  "Storyline",      "06_storyline.py",      "build_storyline"),
    ]

    t0 = datetime.now()
    for idx, name, fname, func in steps:
        if idx in skip:
            print(f"[{idx}] {name} — skipped")
            continue
        print(f"\n[{idx}] {name}")
        mod = _import(fname)
        getattr(mod, func)(ctx)

    print("\n[7] Metadata")
    build_metadata(ctx)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Results → {RESULTS_DIR}")
    print(f"\nRun viewer:")
    print(f"  streamlit run {Path(__file__).parent.parent / 'app.py'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
