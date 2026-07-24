"""loader.py — Checkpoint loading & edge prediction (BCE + edge-level split)"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if not hasattr(np, '_core'):
    import numpy.core as _nc
    sys.modules['numpy._core'] = _nc
    sys.modules['numpy._core.multiarray'] = _nc.multiarray
    sys.modules['numpy._core.umath'] = _nc.umath

from .model import SimpleCatHeteroGAT

_DEFAULT_DIR = Path(__file__).parent.parent.parent  # data_266/


def load_state(ckpt_dir: Path = _DEFAULT_DIR):
    """
    BCE + edge-level split 체크포인트 로드.

    Returns
    -------
    dict with keys:
        model, data, h_emb (np.ndarray, raw 64d),
        ckpt, device,
        pat_list, co_list, ipc_list,
        pat_to_idx, ipc_to_idx, co_to_idx,
        sued_edges, train_edges, val_edges, test_edges,
        best_threshold,
        df_cit, df_ipc, df_asgn,
        tv_plt_set, tv_def_set   ← train sued 엣지 기준 원고/피고
    """
    ckpt_path = ckpt_dir / "phase3_simplecat_ckpt.pt"
    data_path = ckpt_dir / "hetero_data.pt"
    assert ckpt_path.exists(), f"체크포인트 없음: {ckpt_path}"
    assert data_path.exists(), f"HeteroData 없음: {data_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    data   = torch.load(data_path, map_location="cpu", weights_only=False).to(device)

    model = SimpleCatHeteroGAT(n_ipc=len(ckpt["ipc_list"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        h_emb = model(data)["patent"].cpu().numpy()   # raw (not L2-normalized)

    sued_e  = np.array(ckpt["sued_edges"])            # (2, N_total)
    tr_idx  = np.array(ckpt["train_edge_idx"])
    va_idx  = np.array(ckpt["val_edge_idx"])
    te_idx  = np.array(ckpt["test_edge_idx"])

    sued_cases = ckpt["sued_cases"]
    if "test_cases" not in ckpt:
        ckpt["test_cases"] = [sued_cases[i] for i in te_idx.tolist()]

    train_edges = sued_e[:, tr_idx]
    val_edges   = sued_e[:, va_idx]
    test_edges  = sued_e[:, te_idx]

    tv_plt_set = set(train_edges[0].tolist()) | set(val_edges[0].tolist())
    tv_def_set = set(train_edges[1].tolist()) | set(val_edges[1].tolist())
    # transductive: 모든 노드가 학습 그래프에 포함 → test 엣지 참여 노드도 galaxy에
    te_plt_set = set(test_edges[0].tolist())
    te_def_set = set(test_edges[1].tolist())
    galaxy_pat_set = tv_plt_set | tv_def_set | te_plt_set | te_def_set
    test_pat_set   = te_plt_set | te_def_set

    best_threshold = float(ckpt.get("best_threshold", 0.5))

    return {
        "model":          model,
        "data":           data,
        "h_emb":          h_emb,
        "ckpt":           ckpt,
        "device":         device,
        "pat_list":       ckpt["pat_list"],
        "co_list":        ckpt["co_list"],
        "ipc_list":       ckpt["ipc_list"],
        "pat_to_idx":     ckpt["pat_to_idx"],
        "ipc_to_idx":     ckpt["ipc_to_idx"],
        "co_to_idx":      ckpt["co_to_idx"],
        "df_cit":         ckpt["df_cit"],
        "df_ipc":         ckpt["df_ipc"],
        "df_asgn":        ckpt["df_asgn"],
        "sued_edges":     sued_e,
        "train_edges":    train_edges,
        "val_edges":      val_edges,
        "test_edges":     test_edges,
        "tv_plt_set":     tv_plt_set,
        "tv_def_set":     tv_def_set,
        "galaxy_pat_set": galaxy_pat_set,
        "test_pat_set":   test_pat_set,
        "best_threshold": best_threshold,
    }


def predict_edges(h_emb: np.ndarray, query_idx: int,
                  threshold: float, exclude_self: bool = True):
    """
    threshold 이상의 probability를 가진 엣지만 반환.
    Returns: list of (patent_idx, probability)  내림차순 정렬
    """
    logits = h_emb @ h_emb[query_idx]          # raw dot product
    if exclude_self:
        logits[query_idx] = -np.inf
    probs = 1.0 / (1.0 + np.exp(-logits))      # sigmoid

    pred_idx = np.where(probs >= threshold)[0]
    result   = [(int(i), float(probs[i])) for i in pred_idx]
    result.sort(key=lambda x: -x[1])
    return result


def predict_top_k(h_emb: np.ndarray, query_idx: int, k: int = 10):
    """
    threshold 무관하게 상위 k개 반환 (확률 내림차순).
    viz_app fallback용.
    """
    logits = h_emb @ h_emb[query_idx]
    logits[query_idx] = -np.inf
    probs  = 1.0 / (1.0 + np.exp(-logits))
    top    = np.argsort(-probs)[:k]
    return [(int(i), float(probs[i])) for i in top]
