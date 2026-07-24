from __future__ import annotations
"""app.py — FastAPI backend for Patent Litigation Galaxy

Run:
    cd src/data_266/viz_app
    uvicorn backend.app:app --reload --port 8000
"""
import warnings
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .loader import load_state, predict_top_k

warnings.filterwarnings("ignore")

_STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model & checkpoint...")
    _STATE.update(load_state())
    print(f"Ready — {len(_STATE['pat_list'])} patents on "
          f"{_STATE['device']}  |  "
          f"Galaxy={len(_STATE['galaxy_pat_set'])}  "
          f"Waiting={len(_STATE['test_pat_set'])}")
    yield
    _STATE.clear()


app = FastAPI(title="Patent Litigation Galaxy API", version="2.0",
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":  "ok",
        "patents": len(_STATE.get("pat_list", [])),
        "galaxy":  len(_STATE.get("galaxy_pat_set", set())),
        "waiting": len(_STATE.get("test_pat_set",   set())),
    }


# ── Patent Lists ──────────────────────────────────────────
@app.get("/patents/galaxy")
def list_galaxy():
    """Galaxy(Train+Val) 특허 목록"""
    pat_list = _STATE["pat_list"]
    tv_plt   = _STATE["tv_plt_set"]
    tv_def   = _STATE["tv_def_set"]
    result = []
    for idx in sorted(_STATE["galaxy_pat_set"]):
        roles = []
        if idx in tv_plt: roles.append("P")
        if idx in tv_def: roles.append("D")
        result.append({"idx": idx, "patent_id": str(pat_list[idx]),
                        "roles": roles})
    return {"patents": result}


@app.get("/patents/waiting")
def list_waiting():
    """Test 특허 대기실 목록"""
    pat_list  = _STATE["pat_list"]
    test_info = _STATE["test_info"]
    result = []
    for idx in sorted(_STATE["test_pat_set"]):
        info    = test_info.get(idx, {})
        roles   = sorted(info.get("roles", set()))
        n_cases = len(info.get("cases",   set()))
        result.append({
            "idx":       idx,
            "patent_id": str(pat_list[idx]),
            "roles":     roles,
            "n_cases":   n_cases,
        })
    return {"patents": result}


# ── Predict (inject simulation) ───────────────────────────
class PredictResponse(BaseModel):
    test_patent_id: str
    test_idx:       int
    roles:          list[str]
    predictions:    list[dict]   # [{rank, patent_id, idx, cosine, risk}]


@app.get("/predict/{patent_id}", response_model=PredictResponse)
def predict(patent_id: str, k: int = 7):
    """
    Test 특허를 갤럭시에 주입했을 때 소송 위험이 높은 상위 K개 반환.
    """
    p2i = _STATE["pat_to_idx"]
    if patent_id not in p2i:
        raise HTTPException(404, f"Patent '{patent_id}' not found")

    idx = p2i[patent_id]
    if idx not in _STATE["test_pat_set"]:
        raise HTTPException(400, f"'{patent_id}' is not a test (waiting-room) patent")

    preds = predict_top_k(
        _STATE["h_emb"], idx,
        exclude_set=_STATE["test_pat_set"],
        k=k,
    )
    pat_list = _STATE["pat_list"]
    info     = _STATE["test_info"].get(idx, {})

    predictions = []
    for rank, (di, cos) in enumerate(preds, 1):
        risk = "HIGH" if cos > 0.75 else "MED" if cos > 0.55 else "LOW"
        predictions.append({
            "rank":       rank,
            "patent_id":  str(pat_list[di]),
            "idx":        di,
            "cosine":     round(cos, 6),
            "risk":       risk,
        })

    return PredictResponse(
        test_patent_id = patent_id,
        test_idx       = idx,
        roles          = sorted(info.get("roles", set())),
        predictions    = predictions,
    )


# ── Graph Data (for custom renderers) ────────────────────
@app.get("/graph/galaxy")
def graph_galaxy():
    """
    Galaxy 그래프의 노드·엣지 JSON (PyVis 없이 직접 렌더링하거나
    D3.js 등에 넘길 때 사용).
    """
    pat_list   = _STATE["pat_list"]
    co_list    = _STATE["co_list"]
    ipc_list   = _STATE["ipc_list"]
    pat_to_idx = _STATE["pat_to_idx"]
    ipc_to_idx = _STATE["ipc_to_idx"]
    co_to_idx  = _STATE["co_to_idx"]
    df_cit     = _STATE["df_cit"]
    df_ipc     = _STATE["df_ipc"]
    df_asgn    = _STATE["df_asgn"]
    tv_plt     = _STATE["tv_plt_set"]
    tv_def     = _STATE["tv_def_set"]
    galaxy     = _STATE["galaxy_pat_set"]
    litigated  = tv_plt | tv_def

    nodes, edges = [], []

    for idx in galaxy:
        pid  = str(pat_list[idx])
        role = ("P&D" if (idx in tv_plt and idx in tv_def)
                else "P" if idx in tv_plt
                else "D" if idx in tv_def
                else "")
        nodes.append({"id": f"p_{idx}", "type": "patent",
                      "label": pid, "role": role})

    seen_e = set()
    for _, row in df_cit.iterrows():
        si = pat_to_idx.get(str(row.get("citing_patent", row.get("patent_id", ""))))
        di = pat_to_idx.get(str(row.get("cited_patent",  row.get("cited_id",  ""))))
        if si is None or di is None: continue
        if si not in galaxy or di not in galaxy: continue
        if (si, di) in seen_e: continue
        seen_e.add((si, di))
        edges.append({"from": f"p_{si}", "to": f"p_{di}", "rel": "cites"})

    ipc_added = set()
    for _, row in df_ipc.iterrows():
        pi = pat_to_idx.get(str(row.get("patent_id", "")))
        ci = ipc_to_idx.get(str(row.get("ipc_code",  "")))
        if pi is None or ci is None: continue
        if pi not in litigated or pi not in galaxy: continue
        if ci not in ipc_added:
            ipc_added.add(ci)
            nodes.append({"id": f"i_{ci}", "type": "ipc",
                          "label": str(ipc_list[ci]), "role": ""})
        edges.append({"from": f"p_{pi}", "to": f"i_{ci}", "rel": "has_ipc"})

    co_added = set()
    for _, row in df_asgn.iterrows():
        pi   = pat_to_idx.get(str(row.get("patent_id", "")))
        co_s = str(row.get("assignee", row.get("company", "")))
        ci2  = co_to_idx.get(co_s)
        if pi is None or ci2 is None: continue
        if pi not in litigated or pi not in galaxy: continue
        if ci2 not in co_added:
            co_added.add(ci2)
            nodes.append({"id": f"co_{ci2}", "type": "company",
                          "label": str(co_list[ci2]), "role": ""})
        edges.append({"from": f"co_{ci2}", "to": f"p_{pi}", "rel": "owns"})

    return {"nodes": nodes, "edges": edges,
            "stats": {"n_patents": len(galaxy),
                      "n_ipc": len(ipc_added),
                      "n_companies": len(co_added),
                      "n_edges": len(edges)}}
