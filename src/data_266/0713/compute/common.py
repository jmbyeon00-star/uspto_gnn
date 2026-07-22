"""compute/common.py — Shared context loader for all XAI compute scripts."""
import sys, json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent.parent.parent.resolve()   # data_266/
RESULTS_DIR = Path(__file__).parent.parent / "results"        # 0713/results/
sys.path.insert(0, str(DATA_DIR / "viz_app"))

from backend.loader import load_state
from backend.model import EDGE_TYPES, _CONV_ACTS

# Unified relation labels (8 raw types → 4 logical)
RELATION_MAP = {
    "cites":       "citation",
    "cited_by":    "citation",
    "owned_by":    "owner",
    "owns":        "owner",
    "has_ipc":     "ipc",
    "rev_has_ipc": "ipc",
    "sued":        "litigation",
    "sued_by":     "litigation",
}

def save_json(data: dict, filename: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / filename
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {filename}")


def _get_labels(df, pat_label, col_pat=0, col_val=1, limit=5):
    if df is None:
        return []
    try:
        col = df.columns[col_pat]
        rows = df[df[col] == pat_label]
        return [str(v) for v in rows.iloc[:, col_val].tolist()][:limit]
    except Exception:
        return []


def sigmoid(x):
    return float(1 / (1 + np.exp(-x)))


def load_context(pair: int = 0, khop: int = 2, topk: int = 10) -> dict:
    """Load model + data and build shared computation context."""
    print(f"\n[common] Loading checkpoint (pair={pair}, khop={khop}, topk={topk})...")
    state = load_state(DATA_DIR)

    model      = state["model"]
    data       = state["data"]
    device     = state["device"]
    test_edges = state["test_edges"]
    pat_list   = state["pat_list"]
    co_list    = state["co_list"]
    ipc_list   = state["ipc_list"]

    src_idx = int(test_edges[0, pair])
    dst_idx = int(test_edges[1, pair])
    src_label = pat_list[src_idx]
    dst_label = pat_list[dst_idx]

    # Run forward pass to get 64-d embeddings
    model.eval()
    with torch.no_grad():
        h_dict = model(data)
        h_emb  = h_dict["patent"].cpu().numpy()   # (N_pat, 64)

    p_base = sigmoid(h_emb[src_idx] @ h_emb[dst_idx])
    print(f"[common] {src_label} → {dst_label}   P(sue)={p_base:.4f}")

    df_asgn = state.get("df_asgn")
    df_ipc  = state.get("df_ipc")
    df_cit  = state.get("df_cit")

    src_owners = _get_labels(df_asgn, src_label)
    dst_owners = _get_labels(df_asgn, dst_label)
    src_ipc    = _get_labels(df_ipc,  src_label)
    dst_ipc    = _get_labels(df_ipc,  dst_label)

    return {
        "model":      model,
        "data":       data,
        "device":     device,
        "h_emb":      h_emb,
        "pair":       pair,
        "src_idx":    src_idx,
        "dst_idx":    dst_idx,
        "src_label":  src_label,
        "dst_label":  dst_label,
        "src_owners": src_owners,
        "dst_owners": dst_owners,
        "src_ipc":    src_ipc,
        "dst_ipc":    dst_ipc,
        "pat_list":   pat_list,
        "co_list":    co_list,
        "ipc_list":   ipc_list,
        "p_base":     p_base,
        "khop":       khop,
        "topk":       topk,
        "df_asgn":    df_asgn,
        "df_ipc":     df_ipc,
        "df_cit":     df_cit,
        "state":      state,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
