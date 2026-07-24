from __future__ import annotations
import streamlit as st

st.set_page_config(
    page_title="Patent Litigation Galaxy",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 나머지 imports (set_page_config 이후) ──────────────────
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit.components.v1 as components
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from pyvis.network import Network
from torch_geometric.utils import k_hop_subgraph
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.loader import load_state, predict_edges, predict_top_k
from backend.model import SimpleCatHeteroGAT, EDGE_TYPES

# ── CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
  html, body, [class*="css"] {
    background-color: #050810 !important;
    color: #FFFFFF;
    font-family: 'JetBrains Mono', monospace;
  }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070B18, #0A0E1A);
    border-right: 1px solid rgba(0,255,255,0.12);
  }
  .glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(14px);
    margin-bottom: 14px;
    transition: border-color .25s;
  }
  .glass-card:hover { border-color: rgba(0,255,255,0.3); }
  .metric-label {
    font-size:.72rem; color:#666;
    letter-spacing:.12em; text-transform:uppercase; margin-bottom:4px;
  }
  .metric-cyan {
    font-size:2.2rem; font-weight:700;
    background: linear-gradient(135deg,#00FFFF,#00BFFF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .metric-gold { font-size:2.2rem; font-weight:700; color:#FFD700; }
  .wait-item {
    background: rgba(0,255,255,0.03);
    border: 1px solid rgba(0,255,255,0.12);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 6px;
  }
  .pid-tag { font-size:.82rem; font-weight:700; color:#00FFFF; }
  .role-p  { font-size:.68rem; color:#FFD700; border:1px solid #FFD70055;
             border-radius:4px; padding:1px 7px; margin-right:4px; }
  .role-d  { font-size:.68rem; color:#FF6B35; border:1px solid #FF6B3555;
             border-radius:4px; padding:1px 7px; }
  .stButton > button {
    background: linear-gradient(135deg,#00FFFF18,#00BFFF18);
    border: 1px solid #00FFFF55; color:#00FFFF;
    border-radius:8px; font-family:'JetBrains Mono',monospace;
    font-weight:700; transition:all .2s;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg,#00FFFF33,#00BFFF33);
    border-color:#00FFFF; box-shadow:0 0 14px #00FFFF33;
  }
  hr { border-color:rgba(255,255,255,0.06) !important; }
  ::-webkit-scrollbar { width:5px; }
  ::-webkit-scrollbar-track { background:#050810; }
  ::-webkit-scrollbar-thumb { background:#00FFFF33; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ── Physics ────────────────────────────────────────────────
_GALAXY_OPTS = json.dumps({
    "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
            "gravitationalConstant": -55, "centralGravity": 0.003,
            "springLength": 120, "springConstant": 0.04,
            "damping": 0.42, "avoidOverlap": 0.6,
        },
        # 사전 계산 후 표시 — 1300+ 노드는 브라우저 실시간 불가
        "stabilization": {"iterations": 280, "updateInterval": 80, "fit": True},
        "maxVelocity": 60, "minVelocity": 0.8,
    },
    "interaction": {"hover": True, "tooltipDelay": 80,
                    "hideEdgesOnDrag": True, "navigationButtons": True},
    "nodes": {"shape": "dot"},
    "edges": {"smooth": {"type": "continuous"}},
})

# test 주입 뷰: updateInterval 낮춰서 애니메이션 진행 보이게
_INJECT_OPTS = json.dumps({
    "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
            "gravitationalConstant": -55, "centralGravity": 0.003,
            "springLength": 120, "springConstant": 0.04,
            "damping": 0.42, "avoidOverlap": 0.6,
        },
        "stabilization": {"iterations": 300, "updateInterval": 30, "fit": True},
        "maxVelocity": 80, "minVelocity": 0.5,
    },
    "interaction": {"hover": True, "tooltipDelay": 80,
                    "hideEdgesOnDrag": True, "navigationButtons": True},
    "nodes": {"shape": "dot"},
    "edges": {"smooth": {"type": "continuous"}},
})

_SUBGRAPH_OPTS = json.dumps({
    "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
            "gravitationalConstant": -80, "centralGravity": 0.01,
            "springLength": 160, "springConstant": 0.07,
            "damping": 0.45, "avoidOverlap": 1.0,
        },
        "stabilization": {"iterations": 180, "updateInterval": 30, "fit": True},
        "maxVelocity": 80, "minVelocity": 1.0,
    },
    "interaction": {"hover": True, "tooltipDelay": 60,
                    "hideEdgesOnDrag": False, "navigationButtons": True},
    "nodes": {"shape": "dot"},
    "edges": {"smooth": {"type": "curvedCW", "roundness": 0.2}},
})


# ══════════════════════════════════════════════════════════
#  ① st.cache_resource — 모델·데이터 RAM 박제 (앱 생애 1회)
# ══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🌌 모델 & 그래프 로딩 중... (최초 1회)")
def _load():
    return load_state()


# ══════════════════════════════════════════════════════════
#  ② 기본 Galaxy HTML 캐시 (session_state — 탭 생애 1회)
#     전체 1,000+노드 렌더링은 최초 한 번만, 이후 재사용
# ══════════════════════════════════════════════════════════
@st.cache_data(show_spinner="🌌 갤럭시 초기 빌드 중... (최초 1회)")
def _build_base_galaxy_html(_state_key: int) -> str:
    """
    전체 Galaxy(Train+Val) PyVis HTML을 한 번 생성해 캐시.
    _state_key는 캐시 키용 더미 (항상 0).
    """
    state    = _load()
    gal      = state["galaxy_pat_set"]
    tv_plt   = state["tv_plt_set"]
    tv_def   = state["tv_def_set"]
    lit      = tv_plt | tv_def
    pl       = state["pat_list"]
    p2i      = state["pat_to_idx"]
    i2i      = state["ipc_to_idx"]
    c2i      = state["co_to_idx"]

    net = Network(height="820px", width="100%", bgcolor="#050810",
                  font_color="#FFFFFF", directed=True,
                  notebook=False, cdn_resources="in_line")

    for idx in gal:
        pid = str(pl[idx])
        is_p, is_d = idx in tv_plt, idx in tv_def
        if   is_p and is_d: color, sz = "#FFA500", 26
        elif is_p:           color, sz = "#FFD700", 22
        elif is_d:           color, sz = "#FF6B35", 22
        else:                color, sz = "#007B9E", 12
        role = ("P&D" if (is_p and is_d)
                else "Plaintiff" if is_p
                else "Defendant" if is_d
                else "•")
        net.add_node(f"p_{idx}", label=pid,
            title=(f"<b>[PATENT]</b><br>{pid}<br>"
                   f"<span style='color:{color}'>◉ {role}</span>"),
            color={"background": color, "border": color,
                   "highlight": {"background": "#FFF", "border": "#FFF"},
                   "hover":     {"background": "#FFF", "border": "#FFF"}},
            size=sz,
            font={"color": "#444" if role == "•" else "#FFF",
                  "size":  9    if role == "•" else 11},
            borderWidth=1,
            shadow={"enabled": True, "color": color + "44",
                    "size": 11 if role != "•" else 4, "x": 0, "y": 0})

    # ── 소송 엣지 (Train+Val sued edges) ──────────────────────
    sued_e = state["ckpt"]["sued_edges"]          # (2, N) or list
    sued_c = state["ckpt"]["sued_cases"]
    test_cs = set(state["ckpt"]["test_cases"])
    import numpy as _np
    sued_arr = _np.array(sued_e)
    seen_sued: set = set()
    for i, case_id in enumerate(sued_c):
        if case_id in test_cs:
            continue
        s, d = int(sued_arr[0, i]), int(sued_arr[1, i])
        if s not in gal or d not in gal: continue
        if (s, d) in seen_sued: continue
        seen_sued.add((s, d))
        net.add_edge(f"p_{s}", f"p_{d}",
                     color={"color": "#8B0000", "hover": "#FF0000"},
                     width=2.5,
                     arrows={"to": {"enabled": True, "scaleFactor": 0.6}},
                     smooth={"type": "curvedCW", "roundness": 0.2},
                     title="⚖️ 소송 관계")

    # ── 인용 엣지 ──────────────────────────────────────────────
    seen: set = set()
    for _, row in state["df_cit"].iterrows():
        si = p2i.get(str(row.get("citing_patent", row.get("patent_id", ""))))
        di = p2i.get(str(row.get("cited_patent",  row.get("citation_patent_id", row.get("cited_id", "")))))
        if si is None or di is None: continue
        if si not in gal or di not in gal: continue
        if (si, di) in seen: continue
        seen.add((si, di))
        net.add_edge(f"p_{si}", f"p_{di}",
                     color={"color": "#002A44", "hover": "#00BFFF"},
                     width=0.9,
                     arrows={"to": {"enabled": True, "scaleFactor": 0.35}},
                     smooth={"type": "continuous"})

    ipc_g: dict = {}
    ipc_edges: list = []
    for _, row in state["df_ipc"].iterrows():
        pi = p2i.get(str(row.get("patent_id", "")))
        ci = i2i.get(str(row.get("ipc_code",  "")))
        if pi is None or ci is None: continue
        if pi not in lit or pi not in gal: continue
        ipc_g[ci] = str(row.get("ipc_code", ""))
        ipc_edges.append((pi, ci))
    for ci, ic in ipc_g.items():
        net.add_node(f"i_{ci}", label=ic,
                     title=f"<b>[IPC]</b><br>{ic}",
                     color={"background": "#132200", "border": "#32CD32",
                            "hover": {"background": "#32CD32"}},
                     size=10, shape="diamond",
                     font={"color": "#32CD32", "size": 8})
    for pi, ci in ipc_edges:
        net.add_edge(f"p_{pi}", f"i_{ci}",
                     color={"color": "#0F2200", "hover": "#7FFF00"},
                     width=0.7, arrows={"to": {"enabled": False}},
                     smooth={"type": "continuous"})

    co_g: dict = {}
    co_edges: list = []
    for _, row in state["df_asgn"].iterrows():
        pi   = p2i.get(str(row.get("patent_id", "")))
        co_s = str(row.get("org_norm", row.get("assignee", row.get("company", ""))))
        ci2  = c2i.get(co_s)
        if pi is None or ci2 is None: continue
        if pi not in lit or pi not in gal: continue
        co_g[ci2] = co_s
        co_edges.append((ci2, pi))
    for ci2, co_s in co_g.items():
        lb = co_s[:16] + ".." if len(co_s) > 16 else co_s
        net.add_node(f"co_{ci2}", label=lb,
                     title=f"<b>[COMPANY]</b><br>{co_s}",
                     color={"background": "#1A0030", "border": "#FF00FF",
                            "hover": {"background": "#FF00FF"}},
                     size=13, shape="square",
                     font={"color": "#FF00FF", "size": 8})
    for ci2, pi in co_edges:
        net.add_edge(f"co_{ci2}", f"p_{pi}",
                     color={"color": "#1A0025", "hover": "#FF69B4"},
                     width=0.7,
                     arrows={"to": {"enabled": True, "scaleFactor": 0.3}},
                     smooth={"type": "continuous"})

    net.set_options(_GALAXY_OPTS)
    return net.generate_html()


# ══════════════════════════════════════════════════════════
#  ②-B 전체 Galaxy + Test 주입 뷰 (test 선택 시 메인 뷰)
# ══════════════════════════════════════════════════════════
def _build_full_injection_html(state: dict, test_idx: int,
                               pred_edges: list[tuple[int, float]],
                               gt_set: set | None = None) -> str:
    """
    전체 Galaxy 위에 test 노드 + 예측 엣지를 올린 PyVis HTML.
    physics stabilization 없이 실시간 애니메이션으로 렌더.
    """
    gal    = state["galaxy_pat_set"]
    tv_plt = state["tv_plt_set"]
    tv_def = state["tv_def_set"]
    lit    = tv_plt | tv_def
    pl     = state["pat_list"]
    p2i    = state["pat_to_idx"]
    i2i    = state["ipc_to_idx"]
    c2i    = state["co_to_idx"]

    net = Network(height="820px", width="100%", bgcolor="#050810",
                  font_color="#FFFFFF", directed=True,
                  notebook=False, cdn_resources="in_line")

    # ── Galaxy 특허 노드 ──────────────────────────────────
    pred_set = {di for di, _ in pred_edges}
    _gt = gt_set or set()
    for idx in gal:
        pid  = str(pl[idx])
        is_p, is_d = idx in tv_plt, idx in tv_def
        is_pred = idx in pred_set
        is_gt   = idx in _gt
        if   is_p and is_d: color, sz = "#FFA500", 26
        elif is_p:           color, sz = "#FFD700", 22
        elif is_d:           color, sz = "#FF6B35", 22
        else:                color, sz = "#007B9E", 12
        if is_pred:          sz = max(sz, 30)
        role = ("P&D" if (is_p and is_d) else "Plaintiff" if is_p
                else "Defendant" if is_d else "•")
        border_col = "#00FF7F" if is_gt else ("#FF0066" if is_pred else color)
        bw         = 4 if is_gt else (3 if is_pred else 1)
        shadow_col = "#00FF7FAA" if is_gt else ("#FF006688" if is_pred else color + "44")
        net.add_node(f"p_{idx}", label=pid,
            title=(f"<b>[PATENT]</b><br>{pid}<br>"
                   f"<span style='color:{color}'>◉ {role}</span>"
                   + (f"<br><b style='color:#00FF7F'>✓ Ground Truth</b>" if is_gt else "")
                   + (f"<br><b style='color:#FF0066'>★ 소송 예측 후보</b>" if is_pred else "")),
            color={"background": color, "border": border_col,
                   "highlight": {"background": "#FFF", "border": "#FFF"},
                   "hover":     {"background": "#FFF", "border": "#FFF"}},
            size=sz,
            font={"color": "#555" if role == "•" and not is_pred else "#FFF",
                  "size": 9 if role == "•" else 11},
            borderWidth=bw,
            shadow={"enabled": True, "color": shadow_col,
                    "size": 22 if is_gt else (18 if is_pred else 6), "x": 0, "y": 0})

    # ── Test 노드 (먼저 추가) ─────────────────────────────
    tpid_str = str(pl[test_idx])
    net.add_node(f"p_{test_idx}", label=tpid_str,
        title=f"<b>🛸 [TEST — INCOMING]</b><br>{tpid_str}",
        color={"background": "#FFFFFF", "border": "#00FFFF",
               "highlight": {"background": "#00FFFF", "border": "#FFF"}},
        size=56, font={"color": "#050810", "size": 14, "bold": True},
        borderWidth=4,
        shadow={"enabled": True, "color": "#FFFFFF99", "size": 44, "x": 0, "y": 0})

    # ── 소송 엣지 (train/val) ─────────────────────────────
    sued_e  = state["ckpt"]["sued_edges"]
    sued_c  = state["ckpt"]["sued_cases"]
    test_cs = set(state["ckpt"]["test_cases"])
    import numpy as _np
    sued_arr = _np.array(sued_e)
    seen_sued: set = set()
    for i, case_id in enumerate(sued_c):
        if case_id in test_cs: continue
        s, d = int(sued_arr[0, i]), int(sued_arr[1, i])
        if s not in gal or d not in gal: continue
        if (s, d) in seen_sued: continue
        seen_sued.add((s, d))
        net.add_edge(f"p_{s}", f"p_{d}",
                     color={"color": "#8B0000", "hover": "#FF0000"},
                     width=2.0,
                     arrows={"to": {"enabled": True, "scaleFactor": 0.5}},
                     smooth={"type": "curvedCW", "roundness": 0.2},
                     title="⚖️ 소송 관계")

    # ── 인용 엣지 ─────────────────────────────────────────
    seen: set = set()
    for _, row in state["df_cit"].iterrows():
        si = p2i.get(str(row.get("citing_patent", row.get("patent_id", ""))))
        di = p2i.get(str(row.get("cited_patent",
                     row.get("citation_patent_id", row.get("cited_id", "")))))
        if si is None or di is None: continue
        if si not in gal or di not in gal: continue
        if (si, di) in seen: continue
        seen.add((si, di))
        net.add_edge(f"p_{si}", f"p_{di}",
                     color={"color": "#002A44", "hover": "#00BFFF"},
                     width=0.8,
                     arrows={"to": {"enabled": True, "scaleFactor": 0.3}},
                     smooth={"type": "continuous"})

    # ── IPC 노드 + 엣지 (소송특허 + test) ────────────────
    ipc_g: dict = {}; ipc_e: list = []
    for _, row in state["df_ipc"].iterrows():
        pid_r = str(row.get("patent_id", ""))
        pi    = p2i.get(pid_r)
        ci    = i2i.get(str(row.get("ipc_code", "")))
        if ci is None: continue
        ic    = str(row.get("ipc_code", ""))
        if pi is not None and pi in lit and pi in gal:
            ipc_g[ci] = ic; ipc_e.append((f"p_{pi}", f"i_{ci}"))
        if pid_r == tpid_str:
            ipc_g[ci] = ic; ipc_e.append((f"p_{test_idx}", f"i_{ci}"))
    for ci, ic in ipc_g.items():
        net.add_node(f"i_{ci}", label=ic, title=f"<b>[IPC]</b><br>{ic}",
                     color={"background": "#132200", "border": "#32CD32",
                            "hover": {"background": "#32CD32"}},
                     size=10, shape="diamond",
                     font={"color": "#32CD32", "size": 8})
    for src, dst in ipc_e:
        net.add_edge(src, dst, color={"color": "#0F2200", "hover": "#7FFF00"},
                     width=0.7, arrows={"to": {"enabled": False}},
                     smooth={"type": "continuous"})

    # ── Company 노드 + 엣지 (소송특허 + test) ────────────
    co_g: dict = {}; co_e: list = []
    for _, row in state["df_asgn"].iterrows():
        pid_r = str(row.get("patent_id", ""))
        pi    = p2i.get(pid_r)
        co_s  = str(row.get("org_norm", row.get("assignee", row.get("company", ""))))
        ci2   = c2i.get(co_s)
        if ci2 is None: continue
        if pi is not None and pi in lit and pi in gal:
            co_g[ci2] = co_s; co_e.append((f"co_{ci2}", f"p_{pi}"))
        if pid_r == tpid_str:
            co_g[ci2] = co_s; co_e.append((f"co_{ci2}", f"p_{test_idx}"))
    for ci2, co_s in co_g.items():
        lb = co_s[:16] + ".." if len(co_s) > 16 else co_s
        net.add_node(f"co_{ci2}", label=lb, title=f"<b>[COMPANY]</b><br>{co_s}",
                     color={"background": "#1A0030", "border": "#FF00FF",
                            "hover": {"background": "#FF00FF"}},
                     size=13, shape="square",
                     font={"color": "#FF00FF", "size": 8})
    for src, dst in co_e:
        net.add_edge(src, dst, color={"color": "#1A0025", "hover": "#FF69B4"},
                     width=0.7,
                     arrows={"to": {"enabled": True, "scaleFactor": 0.3}},
                     smooth={"type": "continuous"})

    # ── 예측 엣지 (GT=초록실선, 그외=네온) ──────────────────
    pred_di_set = set()
    if pred_edges:
        mx = max(c for _, c in pred_edges)
        for rank, (di, cos) in enumerate(pred_edges, 1):
            pred_di_set.add(di)
            is_gt_edge = di in _gt
            if is_gt_edge:
                dc = "#00FF7F"
                title_str = f"<b>✓ Ground Truth (적중)</b><br>Cosine: {cos:.4f}  |  Rank: {rank}"
            else:
                dc = ("#FF0066" if cos >= _THR_HIGH
                      else "#FF6600" if cos >= _THR_MED
                      else "#FFAA00")
                title_str = f"<b>⚠️ 소송 예측</b><br>Cosine: {cos:.4f}  |  Rank: {rank}"
            w  = 3.0 + (cos / (mx + 1e-8)) * 12
            net.add_edge(
                f"p_{test_idx}", f"p_{di}",
                label=f"{'✓' if is_gt_edge else '#'}{rank} {cos:.3f}",
                font={"color": dc, "size": 12, "bold": True, "align": "middle"},
                title=title_str,
                width=w,
                color={"color": dc, "highlight": "#FFF", "hover": "#FFF"},
                arrows={"to": {"enabled": False}},
                smooth={"type": "curvedCW", "roundness": 0.25},
                shadow={"enabled": True, "color": dc + "CC", "size": 22, "x": 0, "y": 0},
            )

    # ── 놓친 GT 엣지 (점선 초록) ─────────────────────────────
    for gt_idx in _gt:
        if gt_idx in pred_di_set:
            continue
        net.add_edge(
            f"p_{test_idx}", f"p_{gt_idx}",
            label="✗ GT(miss)",
            font={"color": "#00FF7F", "size": 10, "bold": True, "align": "middle"},
            title="<b>✗ Ground Truth (미예측)</b><br>모델이 놓친 실제 소송 관계",
            width=2.5,
            color={"color": "#00FF7F66", "highlight": "#00FF7F", "hover": "#00FF7F"},
            dashes=True,
            arrows={"to": {"enabled": False}},
            smooth={"type": "curvedCCW", "roundness": 0.3},
            shadow={"enabled": False},
        )

    net.set_options(_INJECT_OPTS)
    return net.generate_html()


# ══════════════════════════════════════════════════════════
#  ③ N-Hop 서브그래프 + Test 노드 주입 (클릭 시 실시간)
# ══════════════════════════════════════════════════════════
def _build_injection_html(state: dict, test_idx: int,
                          pred_edges: list[tuple[int, float]],
                          n_hops: int = 2,
                          gt_set: set | None = None) -> str:
    """
    test_idx 기준 n-hop 이내 galaxy 노드만 오려내고
    test 노드 + 예측 엣지를 추가한 PyVis HTML 반환.
    citation + sued 엣지 모두 hop 탐색에 사용.
    """
    gal    = state["galaxy_pat_set"]
    tv_plt = state["tv_plt_set"]
    tv_def = state["tv_def_set"]
    lit    = tv_plt | tv_def
    pl     = state["pat_list"]
    p2i    = state["pat_to_idx"]
    i2i    = state["ipc_to_idx"]
    c2i    = state["co_to_idx"]
    data   = state["data"]          # HeteroData (GPU/CPU)

    # ── n-hop 서브그래프: test 특허 1개만 앵커, citation + sued 탐색 ──
    anchor_nodes = torch.tensor([test_idx], dtype=torch.long)
    ei_cite = data[("patent", "cites", "patent")].edge_index.cpu()
    import numpy as _np2
    sued_arr2 = _np2.array(state["ckpt"]["sued_edges"])
    ei_sued   = torch.tensor(sued_arr2, dtype=torch.long)
    ei_all    = torch.cat([ei_cite, ei_sued], dim=1)   # citation + lawsuit

    subset, _, _, _ = k_hop_subgraph(
        anchor_nodes, num_hops=n_hops,
        edge_index=ei_all,
        num_nodes=len(pl),
        relabel_nodes=False,
    )
    hop2_set = set(subset.tolist()) & gal   # galaxy 안에 있는 것만

    # ── PyVis 네트워크 ───────────────────────────────────
    net = Network(height="760px", width="100%", bgcolor="#050810",
                  font_color="#FFFFFF", directed=True,
                  notebook=False, cdn_resources="in_line")

    _gt2 = gt_set or set()
    # Patent 노드 (2-hop 서브셋)
    for idx in hop2_set:
        pid  = str(pl[idx])
        is_p = idx in tv_plt
        is_d = idx in tv_def
        is_pred = any(idx == di for di, _ in pred_edges)
        is_gt   = idx in _gt2

        if   is_p and is_d: color, sz = "#FFA500", 26
        elif is_p:           color, sz = "#FFD700", 22
        elif is_d:           color, sz = "#FF6B35", 22
        else:                color, sz = "#007B9E", 14
        if is_pred:          sz = max(sz, 28)

        role = ("P&D" if (is_p and is_d)
                else "Plaintiff" if is_p
                else "Defendant" if is_d
                else "•")
        border_col = "#00FF7F" if is_gt else color
        bw         = 4 if is_gt else 1
        shadow_col = "#00FF7FAA" if is_gt else color + "55"
        net.add_node(f"p_{idx}", label=pid,
            title=(f"<b>[PATENT]</b><br>{pid}<br>"
                   f"<span style='color:{color}'>◉ {role}</span>"
                   + (f"<br><b style='color:#00FF7F'>✓ Ground Truth</b>" if is_gt else "")),
            color={"background": color, "border": border_col,
                   "highlight": {"background": "#FFF", "border": "#FFF"},
                   "hover":     {"background": "#FFF", "border": "#FFF"}},
            size=sz,
            font={"color": "#888" if role == "•" else "#FFF", "size": 10},
            borderWidth=bw,
            shadow={"enabled": True, "color": shadow_col,
                    "size": 20 if is_gt else (12 if role != "•" else 5), "x": 0, "y": 0})

    # Citation 엣지 (서브셋 내부만)
    seen: set = set()
    for _, row in state["df_cit"].iterrows():
        si = p2i.get(str(row.get("citing_patent", row.get("patent_id", ""))))
        di = p2i.get(str(row.get("cited_patent",  row.get("citation_patent_id", row.get("cited_id", "")))))
        if si is None or di is None: continue
        if si not in hop2_set or di not in hop2_set: continue
        if (si, di) in seen: continue
        seen.add((si, di))
        net.add_edge(f"p_{si}", f"p_{di}",
                     color={"color": "#003566", "hover": "#00BFFF"},
                     width=1.2,
                     arrows={"to": {"enabled": True, "scaleFactor": 0.5}},
                     smooth={"type": "continuous"})

    # ── Test 노드 먼저 추가 (IPC/Company 엣지가 참조하기 전에) ──
    tpid_str = str(pl[test_idx])
    net.add_node(f"p_{test_idx}", label=tpid_str,
        title=f"<b>🛸 [TEST — INCOMING]</b><br>{tpid_str}",
        color={"background": "#FFFFFF", "border": "#00FFFF",
               "highlight": {"background": "#00FFFF", "border": "#FFF"}},
        size=56,
        font={"color": "#050810", "size": 14, "bold": True},
        borderWidth=4,
        shadow={"enabled": True, "color": "#FFFFFF99", "size": 44, "x": 0, "y": 0})

    # IPC 노드 → 엣지 순서로 추가 (서브셋 내 소송특허 + test 특허)
    ipc_g: dict = {}
    ipc_edges_sub: list = []
    for _, row in state["df_ipc"].iterrows():
        pid_row = str(row.get("patent_id", ""))
        pi  = p2i.get(pid_row)
        ci  = i2i.get(str(row.get("ipc_code", "")))
        if ci is None: continue
        ipc_code = str(row.get("ipc_code", ""))
        # Galaxy 소송특허 IPC
        if pi is not None and pi in lit and pi in hop2_set:
            ipc_g[ci] = ipc_code
            ipc_edges_sub.append((f"p_{pi}", f"i_{ci}"))
        # Test 특허 자체 IPC
        if pid_row == tpid_str:
            ipc_g[ci] = ipc_code
            ipc_edges_sub.append((f"p_{test_idx}", f"i_{ci}"))
    for ci, ic in ipc_g.items():
        net.add_node(f"i_{ci}", label=ic,
                     title=f"<b>[IPC]</b><br>{ic}",
                     color={"background": "#1A4400", "border": "#32CD32",
                            "hover": {"background": "#32CD32"}},
                     size=12, shape="diamond",
                     font={"color": "#32CD32", "size": 9})
    for src, dst in ipc_edges_sub:
        net.add_edge(src, dst,
                     color={"color": "#1A3300", "hover": "#7FFF00"},
                     width=1.0, arrows={"to": {"enabled": False}},
                     smooth={"type": "continuous"})

    # Company 노드 → 엣지 순서로 추가 (서브셋 내 소송특허 + test 특허)
    co_g: dict = {}
    co_edges_sub: list = []
    for _, row in state["df_asgn"].iterrows():
        pid_row = str(row.get("patent_id", ""))
        pi   = p2i.get(pid_row)
        co_s = str(row.get("org_norm", row.get("assignee", row.get("company", ""))))
        ci2  = c2i.get(co_s)
        if ci2 is None: continue
        # Galaxy 소송특허 company
        if pi is not None and pi in lit and pi in hop2_set:
            co_g[ci2] = co_s
            co_edges_sub.append((f"co_{ci2}", f"p_{pi}"))
        # Test 특허 자체 company
        if pid_row == tpid_str:
            co_g[ci2] = co_s
            co_edges_sub.append((f"co_{ci2}", f"p_{test_idx}"))
    for ci2, co_s in co_g.items():
        lb = co_s[:16] + ".." if len(co_s) > 16 else co_s
        net.add_node(f"co_{ci2}", label=lb,
                     title=f"<b>[COMPANY]</b><br>{co_s}",
                     color={"background": "#1A0030", "border": "#FF00FF",
                            "hover": {"background": "#FF00FF"}},
                     size=14, shape="square",
                     font={"color": "#FF00FF", "size": 9})
    for src, dst in co_edges_sub:
        net.add_edge(src, dst,
                     color={"color": "#2A0035", "hover": "#FF69B4"},
                     width=1.0,
                     arrows={"to": {"enabled": True, "scaleFactor": 0.4}},
                     smooth={"type": "continuous"})

    # ── 예측 엣지 (GT=초록실선, 그외=네온) ──────────────────
    existing_nodes = set(net.get_nodes())
    pred_di_set2 = set()
    if pred_edges:
        mx = max(c for _, c in pred_edges)
        for rank, (di, cos) in enumerate(pred_edges, 1):
            if f"p_{di}" not in existing_nodes:
                continue
            pred_di_set2.add(di)
            is_gt_edge = di in _gt2
            w  = 2.5 + (cos / (mx + 1e-8)) * 13
            if is_gt_edge:
                dc = "#00FF7F"
                title_str = f"<b>✓ Ground Truth (적중)</b><br>Cosine: {cos:.4f}  |  Rank: {rank}"
            else:
                dc = ("#FF0066" if cos > 0.75
                      else "#FF6600" if cos > 0.55
                      else "#FFAA00")
                title_str = f"<b>⚠️ 소송 예측</b><br>Cosine: {cos:.4f}  |  Rank: {rank}"
            net.add_edge(
                f"p_{test_idx}", f"p_{di}",
                label=f"{'✓' if is_gt_edge else '#'}{rank}  {cos:.3f}",
                font={"color": dc, "size": 11, "bold": True, "align": "middle"},
                title=title_str,
                width=w,
                color={"color": dc, "highlight": "#FFF", "hover": "#FFF"},
                arrows={"to": {"enabled": False}},
                smooth={"type": "curvedCW", "roundness": 0.3},
                shadow={"enabled": True, "color": dc + "CC",
                        "size": 20, "x": 0, "y": 0},
            )

    # ── 놓친 GT 엣지 (점선 초록) ─────────────────────────────
    for gt_idx in _gt2:
        if gt_idx in pred_di_set2:
            continue
        if f"p_{gt_idx}" not in existing_nodes:
            continue
        net.add_edge(
            f"p_{test_idx}", f"p_{gt_idx}",
            label="✗ GT(miss)",
            font={"color": "#00FF7F", "size": 10, "bold": True, "align": "middle"},
            title="<b>✗ Ground Truth (미예측)</b><br>모델이 놓친 실제 소송 관계",
            width=2.5,
            color={"color": "#00FF7F66", "highlight": "#00FF7F", "hover": "#00FF7F"},
            dashes=True,
            arrows={"to": {"enabled": False}},
            smooth={"type": "curvedCCW", "roundness": 0.3},
            shadow={"enabled": False},
        )

    net.set_options(_SUBGRAPH_OPTS)
    return net.generate_html()


# ══════════════════════════════════════════════════════════
#  ④ 데이터 기반 임계값 & 헬퍼
# ══════════════════════════════════════════════════════════
# 실제 cosine 분포에서 도출
#   test 소송쌍:   mean=0.66  25th=0.55  75th=0.79
#   random 비소송: mean=0.23  75th=0.44
_THR_HIGH = 0.72   # test 소송쌍 75th → HIGH
_THR_MED  = 0.55   # test 소송쌍 25th → MED
_THR_NONE = 0.45   # random 비소송 75th → 위험 없음 경계


def _perf(preds: list, gt_set: set) -> dict:
    """preds = [(idx, prob), ...] 기준 P/R/F1 계산."""
    n_pred = len(preds)
    hits   = sum(1 for di, _ in preds if di in gt_set)
    n_gt   = len(gt_set)
    prec   = hits / n_pred if n_pred > 0 else 0.0
    rec    = hits / n_gt   if n_gt   > 0 else 0.0
    f1     = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"n_pred": n_pred, "hits": hits, "n_gt": n_gt,
            "prec": prec, "rec": rec, "f1": f1}


@st.cache_data(show_spinner="전체 test 성능 집계 중…")
def _agg_perf(h_emb: np.ndarray, te_arr: np.ndarray,
              thr: float, k: int) -> dict:
    """모든 test 특허에 대해 threshold / top-K 성능 집계."""
    from backend.loader import predict_edges as _pe, predict_top_k as _pk
    rows_thr, rows_topk = [], []
    for src_idx in set(te_arr[0].tolist()) | set(te_arr[1].tolist()):
        gt = (set(te_arr[1, te_arr[0] == src_idx].tolist()) |
              set(te_arr[0, te_arr[1] == src_idx].tolist()))
        if not gt:
            continue
        rows_thr.append(_perf(_pe(h_emb, src_idx, threshold=thr), gt))
        rows_topk.append(_perf(_pk(h_emb, src_idx, k=k), gt))

    def _agg(rows):
        if not rows: return {}
        return {
            "prec":   float(np.mean([r["prec"]  for r in rows])),
            "rec":    float(np.mean([r["rec"]   for r in rows])),
            "f1":     float(np.mean([r["f1"]    for r in rows])),
            "hits":   int(sum(r["hits"]   for r in rows)),
            "n_pred": int(sum(r["n_pred"] for r in rows)),
            "n_gt":   int(sum(r["n_gt"]   for r in rows)),
            "n_pats": len(rows),
        }
    return {"threshold": _agg(rows_thr), "topk": _agg(rows_topk)}


@st.cache_data(show_spinner="t-SNE 계산 중… (최초 1회)")
def _tsne_xy(h_emb: np.ndarray, perplexity: int = 30) -> np.ndarray:
    """h_emb (N, 64) → 2D t-SNE 좌표 (N, 2). 결과는 캐시."""
    xy = TSNE(n_components=2, perplexity=perplexity,
              n_iter=1000, random_state=42).fit_transform(h_emb)
    # 0~1 정규화
    xy -= xy.min(axis=0)
    xy /= xy.max(axis=0) + 1e-8
    return xy


def _build_tsne_fig(xy: np.ndarray, h_emb: np.ndarray, pat_list: list,
                    sel_idx: int, gt_set: set,
                    sued_edges: np.ndarray, thr: float) -> go.Figure:
    """
    t-SNE scatter (Plotly).
    - 색  : P(sued | 선택특허) = sigmoid(h[i] · h[sel]) — 어두운보라=낮음, 밝은노랑=높음
    - 크기: 해당 특허가 소송에 등장한 총 횟수 (sued_edges 기준)
    - ★  : 선택 특허 (흰 별)
    - ◆  : GT 피고 (실제 소송 기록)
         - 흰 테두리 → TP (모델이 예측도 맞음)
         - 테두리 없음 → FN (모델이 놓침)
    """
    N = len(pat_list)

    # ── P(sue) 전체 계산 ──────────────────────────────────────
    logits = h_emb @ h_emb[sel_idx]
    logits[sel_idx] = -np.inf
    probs  = 1.0 / (1.0 + np.exp(-logits))          # (N,)

    # ── 소송 등장 횟수 (원고+피고) ───────────────────────────
    sue_count = np.zeros(N, dtype=int)
    for i in sued_edges[0]: sue_count[i] += 1
    for i in sued_edges[1]: sue_count[i] += 1

    # 마커 크기: 최소 4, 최대 20
    raw_sz  = sue_count.astype(float)
    max_sz  = raw_sz.max() if raw_sz.max() > 0 else 1.0
    marker_sz = 4 + (raw_sz / max_sz) * 16          # (N,)

    # ── 배경 점 (GT / 선택 특허 제외) ───────────────────────
    bg_idx = [i for i in range(N) if i != sel_idx and i not in gt_set]

    fig = go.Figure()

    # 배경 점: 색=P(sue), 크기=소송횟수
    fig.add_trace(go.Scatter(
        x=xy[bg_idx, 0], y=xy[bg_idx, 1],
        mode="markers",
        marker=dict(
            size=[float(marker_sz[i]) for i in bg_idx],
            color=[float(probs[i]) for i in bg_idx],
            colorscale="Plasma",       # 어두운보라→밝은노랑
            cmin=0.0, cmax=1.0,
            showscale=True,
            colorbar=dict(
                title=dict(text="P(sue)", font=dict(color="#888", size=11)),
                tickfont=dict(color="#888", size=10),
                len=0.6, thickness=12,
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0", "0.25", "0.5", "0.75", "1.0"],
            ),
            opacity=0.75,
            line=dict(width=0),
        ),
        text=[f"{pat_list[i]}<br>P(sue)={probs[i]:.3f}<br>소송횟수={sue_count[i]}"
              for i in bg_idx],
        hovertemplate="%{text}<extra></extra>",
        name="전체 특허",
        showlegend=False,
    ))

    # ── GT 피고 (◆ 주황 다이아) — TP/FN 구분, customdata로 클릭 연동 ─
    tp_list = [i for i in gt_set if probs[i] >= thr]
    fn_list = [i for i in gt_set if probs[i] < thr]
    if tp_list:
        fig.add_trace(go.Scatter(
            x=[xy[i, 0] for i in tp_list], y=[xy[i, 1] for i in tp_list],
            mode="markers",
            marker=dict(size=22, symbol="diamond", color="#FF8C00",
                        line=dict(color="#FF0000", width=4)),
            customdata=[[i] for i in tp_list],
            text=[f"{pat_list[i]}<br>P(sue)={probs[i]:.3f}<br>소송횟수={sue_count[i]}"
                  f"<br>✓ TP — 클릭하면 XAI 탭으로 분석" for i in tp_list],
            hovertemplate="%{text}<extra>◆ GT TP</extra>",
            name="◆ GT TP (빨간테두리·클릭→XAI)", showlegend=True,
        ))
    if fn_list:
        fig.add_trace(go.Scatter(
            x=[xy[i, 0] for i in fn_list], y=[xy[i, 1] for i in fn_list],
            mode="markers",
            marker=dict(size=22, symbol="diamond", color="#FF8C00",
                        line=dict(color="#FF8C00", width=0)),
            customdata=[[i] for i in fn_list],
            text=[f"{pat_list[i]}<br>P(sue)={probs[i]:.3f}<br>소송횟수={sue_count[i]}"
                  f"<br>✗ FN — 클릭하면 XAI 탭으로 분석" for i in fn_list],
            hovertemplate="%{text}<extra>◆ GT FN</extra>",
            name="◆ GT FN (테두리없음·클릭→XAI)", showlegend=True,
        ))

    # ── 선택 특허 (★ 흰 별) ──────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[xy[sel_idx, 0]], y=[xy[sel_idx, 1]],
        mode="markers+text",
        marker=dict(size=24, symbol="star", color="#FFFFFF",
                    line=dict(color="#00FFFF", width=3)),
        text=[str(pat_list[sel_idx])],
        textposition="top center",
        textfont=dict(color="#00FFFF", size=11, family="monospace"),
        hovertemplate=(f"{pat_list[sel_idx]}<br>소송횟수={sue_count[sel_idx]}"
                       f"<extra>🛸 선택 특허</extra>"),
        name="🛸 선택 특허",
    ))

    fig.update_layout(
        paper_bgcolor="#050810", plot_bgcolor="#050810",
        font=dict(color="#888"),
        legend=dict(bgcolor="rgba(5,8,16,0.85)", bordercolor="#222",
                    borderwidth=1, font=dict(size=11), x=0.01, y=0.01),
        margin=dict(l=0, r=0, t=36, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=580,
        title=dict(
            text=("t-SNE — GAT 임베딩 공간  "
                  "<span style='font-size:11px;color:#666'>"
                  "색=P(sue 선택특허 기준) · 크기=소송 등장 횟수 · "
                  "◆주황=실제GT (흰테두리=적중)</span>"),
            font=dict(color="#00FFFF", size=13), x=0.01),
    )
    return fig


def _risk(cos: float) -> tuple:
    """(label, color, pct) 반환."""
    if cos >= _THR_HIGH:
        pct = min(100, int((cos - _THR_HIGH) / (1.0 - _THR_HIGH) * 30 + 70))
        return "HIGH", "#FF0066", pct
    if cos >= _THR_MED:
        pct = int((cos - _THR_MED) / (_THR_HIGH - _THR_MED) * 30 + 40)
        return "MED", "#FF6600", pct
    if cos >= _THR_NONE:
        pct = int((cos - _THR_NONE) / (_THR_MED - _THR_NONE) * 30 + 10)
        return "LOW", "#FFAA00", pct
    return "없음", "#444444", 0


def _shared_features(state: dict, test_idx: int, target_idx: int) -> dict:
    """test ↔ target 공통 IPC/인용/회사 집합 반환."""
    pl       = state["pat_list"]
    test_pid = str(pl[test_idx])
    tgt_pid  = str(pl[target_idx])
    df_ipc   = state["df_ipc"]
    df_cit   = state["df_cit"]
    df_asgn  = state["df_asgn"]

    t_ipc = set(df_ipc[df_ipc.patent_id == test_pid]["ipc_code"].astype(str))
    g_ipc = set(df_ipc[df_ipc.patent_id == tgt_pid ]["ipc_code"].astype(str))

    t_cit = set(df_cit[df_cit.patent_id == test_pid]["citation_patent_id"].astype(str))
    g_cit = set(df_cit[df_cit.patent_id == tgt_pid ]["citation_patent_id"].astype(str))

    t_co  = set(df_asgn[df_asgn.patent_id == test_pid]["org_norm"].astype(str))
    g_co  = set(df_asgn[df_asgn.patent_id == tgt_pid ]["org_norm"].astype(str))

    return {
        "ipc":    t_ipc & g_ipc,
        "cit":    t_cit & g_cit,
        "co":     t_co  & g_co,
        "direct": tgt_pid in t_cit or test_pid in g_cit,
    }


def _build_reason_graph_html(state: dict, test_idx: int,
                             target_idx: int, cos: float) -> str:
    """
    test ↔ target 사이 공통 IPC/회사/인용을 PyVis 미니 그래프로 시각화.
    """
    pl  = state["pat_list"]
    sf  = _shared_features(state, test_idx, target_idx)
    _, col, _ = _risk(cos)

    test_pid   = str(pl[test_idx])
    target_pid = str(pl[target_idx])

    net = Network(height="300px", width="100%", bgcolor="#050810",
                  font_color="#FFFFFF", directed=True,
                  notebook=False, cdn_resources="in_line")

    net.add_node("test", label=test_pid,
        title=f"<b>🛸 TEST</b><br>{test_pid}",
        color={"background": "#FFFFFF", "border": "#00FFFF",
               "highlight": {"background": "#00FFFF"}},
        size=30, font={"color": "#050810", "size": 11, "bold": True},
        borderWidth=3,
        shadow={"enabled": True, "color": "#FFFFFF88", "size": 20, "x": 0, "y": 0})

    net.add_node("target", label=target_pid,
        title=(f"<b>⚠️ 소송 후보</b><br>{target_pid}"
               f"<br>Cosine: {cos:.4f}"),
        color={"background": col, "border": col,
               "highlight": {"background": "#FFF"}},
        size=24, font={"color": "#FFF", "size": 11},
        shadow={"enabled": True, "color": col + "88", "size": 14, "x": 0, "y": 0})

    if sf["direct"]:
        net.add_edge("test", "target",
            label="직접 인용",
            color={"color": "#00BFFF", "hover": "#FFF"}, width=2.5,
            arrows={"to": {"enabled": True, "scaleFactor": 0.7}},
            font={"color": "#00BFFF", "size": 9, "bold": True, "align": "middle"})

    for ipc in sorted(sf["ipc"])[:5]:
        nid = f"ipc_{ipc}"
        net.add_node(nid, label=ipc,
            title=f"<b>[공통 IPC]</b><br>{ipc}",
            color={"background": "#132200", "border": "#32CD32",
                   "hover": {"background": "#32CD32"}},
            size=12, shape="diamond",
            font={"color": "#32CD32", "size": 8})
        net.add_edge("test",   nid, color={"color": "#1A4400", "hover": "#7FFF00"},
                     width=0.9, arrows={"to": {"enabled": False}})
        net.add_edge("target", nid, color={"color": "#1A4400", "hover": "#7FFF00"},
                     width=0.9, arrows={"to": {"enabled": False}})

    for co in sorted(sf["co"])[:3]:
        nid = f"co_{co[:20]}"
        lb  = co[:14] + ".." if len(co) > 14 else co
        net.add_node(nid, label=lb,
            title=f"<b>[공통 출원인]</b><br>{co}",
            color={"background": "#1A0030", "border": "#FF00FF",
                   "hover": {"background": "#FF00FF"}},
            size=12, shape="square",
            font={"color": "#FF00FF", "size": 8})
        net.add_edge("test",   nid, color={"color": "#2A0040", "hover": "#FF69B4"},
                     width=0.9, arrows={"to": {"enabled": False}})
        net.add_edge("target", nid, color={"color": "#2A0040", "hover": "#FF69B4"},
                     width=0.9, arrows={"to": {"enabled": False}})

    for cit in sorted(sf["cit"])[:3]:
        nid = f"cit_{cit}"
        net.add_node(nid, label=cit[:12],
            title=f"<b>[공동 인용 선행기술]</b><br>{cit}",
            color={"background": "#001A33", "border": "#00BFFF",
                   "hover": {"background": "#00BFFF"}},
            size=9, shape="triangle",
            font={"color": "#00BFFF", "size": 7})
        net.add_edge("test",   nid, color={"color": "#001A33", "hover": "#00BFFF"},
                     width=0.6, arrows={"to": {"enabled": False}})
        net.add_edge("target", nid, color={"color": "#001A33", "hover": "#00BFFF"},
                     width=0.6, arrows={"to": {"enabled": False}})

    if not any([sf["direct"], sf["ipc"], sf["co"], sf["cit"]]):
        net.add_edge("test", "target",
            label=f"임베딩 유사도 {cos:.3f}",
            color={"color": "#666"}, width=1.5, dashes=True,
            font={"color": "#888", "size": 9, "align": "middle"})

    net.set_options(json.dumps({
        "physics": {
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -150, "centralGravity": 0.02,
                "springLength": 110, "springConstant": 0.12,
                "damping": 0.55, "avoidOverlap": 1.2,
            },
            "stabilization": {"iterations": 120, "fit": True},
        },
        "interaction": {"hover": True, "tooltipDelay": 50, "navigationButtons": False},
        "nodes": {"shape": "dot"},
        "edges": {"smooth": {"type": "continuous"}},
    }))
    return net.generate_html()


# ══════════════════════════════════════════════════════════
#  XAI 유틸 & 계산 함수
# ══════════════════════════════════════════════════════════
def _node_label(idx, ntype, state):
    pl = state["pat_list"]; il = state["ipc_list"]; cl = state["co_list"]
    if ntype == 'patent': return str(pl[idx])
    if ntype == 'ipc':    return il[idx] if idx < len(il) else f'IPC{idx}'
    return cl[idx] if idx < len(cl) else f'CO{idx}'

def _get_patent_neighbors(node_idx, state):
    nb = set()
    d = state["data"]
    for et in EDGE_TYPES:
        src_t, _, dst_t = et
        if et not in d.edge_index_dict: continue
        eidx = d.edge_index_dict[et].cpu().numpy()
        if src_t == 'patent' and dst_t == 'patent':
            nb.update(eidx[1][eidx[0] == node_idx].tolist())
            nb.update(eidx[0][eidx[1] == node_idx].tolist())
    return nb - {node_idx}

class _PGEdgeMLP(nn.Module):
    def __init__(self, emb_dim=64, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, 1))
    def forward(self, hi, hj):
        return self.net(torch.cat([hi, hj], dim=-1)).squeeze(-1)

def _compute_attention(src_idx, dst_idx, state):
    model = state["model"]; data = state["data"]
    x_raw = torch.cat([data['patent'].struct, data['patent'].x], dim=1)
    x = {
        'patent':  model.pat_proj(x_raw),
        'company': F.leaky_relu(model.co_proj(data['company'].x)),
        'ipc':     model.ipc_emb.weight.clone(),
    }
    acts = [F.leaky_relu, F.leaky_relu, torch.tanh]
    all_attn = {}
    for li, (hconv, bn, act) in enumerate(zip(model.convs[:3], model.bns, acts)):
        x_agg = defaultdict(list); layer_attn = {}
        for et in EDGE_TYPES:
            src_t, rel, dst_t = et
            if et not in data.edge_index_dict or et not in hconv.convs: continue
            et_conv = hconv.convs[et]; edge_idx = data.edge_index_dict[et]
            with torch.no_grad():
                out, (ei_ret, alpha) = et_conv(
                    (x[src_t], x[dst_t]), edge_idx, return_attention_weights=True)
            for target_idx in [src_idx, dst_idx]:
                if dst_t == 'patent':
                    mask = (ei_ret[1] == target_idx)
                    if mask.any():
                        layer_attn.setdefault(target_idx, {})[et] = {
                            'src': ei_ret[0][mask].cpu().tolist(),
                            'attn': alpha[mask].mean(dim=1).cpu().tolist()}
            x_agg[dst_t].append(out)
        all_attn[f'L{li+1}'] = layer_attn
        x_new = {k: torch.stack(vs).mean(0) for k, vs in x_agg.items()}
        x = {k: model.drop(act(bn[k](v.view(-1, model.hid)))) if k in bn
             else model.drop(act(v)) for k, v in x_new.items()}
    return all_attn

def _compute_pg(src_idx, dst_idx, state, n_epochs=300):
    h_np = state["h_emb"]; sued_e = np.array(state["ckpt"]["sued_edges"])
    h_t  = torch.tensor(h_np, dtype=torch.float32)
    pg   = _PGEdgeMLP(emb_dim=h_np.shape[1])
    opt  = torch.optim.Adam(pg.parameters(), lr=1e-3)
    rng  = np.random.default_rng(42)
    pos_s = torch.tensor(sued_e[0], dtype=torch.long)
    pos_d = torch.tensor(sued_e[1], dtype=torch.long)
    for _ in range(n_epochs):
        opt.zero_grad()
        ns = torch.tensor(rng.integers(0, len(h_np), size=len(pos_s)), dtype=torch.long)
        nd = torch.tensor(rng.integers(0, len(h_np), size=len(pos_s)), dtype=torch.long)
        all_s = torch.cat([pos_s, ns]); all_d = torch.cat([pos_d, nd])
        lbl   = torch.cat([torch.ones(len(pos_s)), torch.zeros(len(ns))])
        F.binary_cross_entropy_with_logits(pg(h_t[all_s], h_t[all_d]), lbl).backward()
        opt.step()
    pg.eval()
    src_nb = _get_patent_neighbors(src_idx, state)
    dst_nb = _get_patent_neighbors(dst_idx, state)
    with torch.no_grad():
        src_sc = {nb: float(torch.sigmoid(pg(h_t[src_idx].unsqueeze(0), h_t[nb].unsqueeze(0)))) for nb in src_nb}
        dst_sc = {nb: float(torch.sigmoid(pg(h_t[dst_idx].unsqueeze(0), h_t[nb].unsqueeze(0)))) for nb in dst_nb}
    return src_sc, dst_sc

def _compute_subgraphx(src_idx, dst_idx, state, n_samples=60):
    model = state["model"]; data = state["data"]
    def pred_masked(masked_nodes):
        ox = data['patent'].x.clone(); os = data['patent'].struct.clone()
        for n in masked_nodes:
            data['patent'].x[n] = 0.0; data['patent'].struct[n] = 0.0
        with torch.no_grad():
            h = model(data)['patent']
            sc = float(torch.sigmoid((h[src_idx] * h[dst_idx]).sum()))
        data['patent'].x = ox; data['patent'].struct = os
        return sc
    cands = list((_get_patent_neighbors(src_idx, state) |
                  _get_patent_neighbors(dst_idx, state)) - {src_idx, dst_idx})[:25]
    rng = np.random.default_rng(42)
    shapley = defaultdict(float); counts = defaultdict(int)
    for _ in range(n_samples):
        perm = rng.permutation(cands); S = set()
        f_S = pred_masked(set(cands) - S)
        for v in perm:
            S.add(v); f_Sv = pred_masked(set(cands) - S)
            shapley[v] += f_Sv - f_S; counts[v] += 1; f_S = f_Sv
    return {v: shapley[v] / counts[v] for v in cands}

def _compute_graphshap(src_idx, dst_idx, state, n_groups=15, n_samples=200):
    from scipy.special import comb as sp_comb
    model = state["model"]; data = state["data"]
    faiss_dim = data['patent'].x.shape[1]
    chunk = faiss_dim // (n_groups - 1)
    groups = [('struct', None, 'struct')]
    for g in range(n_groups - 1):
        s = g * chunk; e = min(s + chunk, faiss_dim)
        groups.append((f'FAISS_{s}-{e}', (s, e), 'faiss'))
    M = len(groups)
    def shap_for_node(target_idx, other_idx):
        ox = data['patent'].x[target_idx].clone()
        os = data['patent'].struct[target_idx].clone()
        rng = np.random.default_rng(42); X_s, Y_s, W_s = [], [], []
        for _ in range(n_samples):
            inc = rng.random(M) > 0.5
            data['patent'].x[target_idx] = ox.clone()
            data['patent'].struct[target_idx] = os.clone()
            for gi, (_, grange, gtype) in enumerate(groups):
                if not inc[gi]:
                    if gtype == 'struct': data['patent'].struct[target_idx] = 0.0
                    else:
                        gs, ge = grange
                        data['patent'].x[target_idx, gs:ge] = 0.0
            with torch.no_grad():
                hm = model(data)['patent']
                sc = float(torch.sigmoid((hm[target_idx] * hm[other_idx]).sum()))
            z = max(1, min(int(inc.sum()), M-1))
            w = (M - 1) / (sp_comb(M, z) * z * (M - z) + 1e-10)
            X_s.append(inc.astype(float)); Y_s.append(sc); W_s.append(w)
        data['patent'].x[target_idx] = ox; data['patent'].struct[target_idx] = os
        X = np.array(X_s); Y = np.array(Y_s); W = np.diag(np.array(W_s))
        A = X.T @ W @ X + 1e-8 * np.eye(M); b = X.T @ (np.diag(W_s) @ Y)
        phi = np.linalg.solve(A, b)
        return {groups[gi][0]: float(phi[gi]) for gi in range(M)}
    return shap_for_node(src_idx, dst_idx), shap_for_node(dst_idx, src_idx)


def _similar_train_cases(state: dict, test_idx: int, target_idx: int, k: int = 3) -> list:
    """(test, target) 쌍과 임베딩이 가장 유사한 훈련 소송 사례 반환 (fallback용)."""
    h       = state["h_emb"]
    ckpt    = state["ckpt"]
    pl      = state["pat_list"]
    sued_e  = np.array(ckpt["sued_edges"])
    sued_c  = ckpt["sued_cases"]
    test_cs = set(ckpt["test_cases"])

    tv_mask = np.array([c not in test_cs for c in sued_c])
    tv_s    = sued_e[0, tv_mask]
    tv_d    = sued_e[1, tv_mask]
    tv_c    = [c for c, m in zip(sued_c, tv_mask) if m]

    query    = h[test_idx] + h[target_idx]
    norm_q   = np.linalg.norm(query) + 1e-8
    query   /= norm_q

    pair_emb = h[tv_s] + h[tv_d]
    norms    = np.linalg.norm(pair_emb, axis=1, keepdims=True) + 1e-8
    pair_emb = pair_emb / norms
    sims     = pair_emb @ query

    top = np.argsort(-sims)[:k]
    return [{"case": tv_c[i], "P": str(pl[tv_s[i]]),
             "D": str(pl[tv_d[i]]), "sim": float(sims[i]),
             "reasons": ["임베딩 유사도 기반"]} for i in top]


def _precompute_pat_features(state: dict) -> dict:
    """df_ipc/df_cit/df_asgn을 patent_id 기준 딕셔너리로 변환 (세션 캐시)."""
    if "_pat_feat_cache" in st.session_state:
        return st.session_state["_pat_feat_cache"]
    df_ipc  = state["df_ipc"]
    df_cit  = state["df_cit"]
    df_asgn = state["df_asgn"]
    ipc_map  = df_ipc.groupby("patent_id")["ipc_code"].apply(set).to_dict()
    cit_map  = df_cit.groupby("patent_id")["citation_patent_id"].apply(set).to_dict()
    asgn_map = df_asgn.groupby("patent_id")["org_norm"].apply(set).to_dict()
    cache = {"ipc": ipc_map, "cit": cit_map, "asgn": asgn_map}
    st.session_state["_pat_feat_cache"] = cache
    return cache


def _structural_similar_cases(state: dict, test_idx: int, target_idx: int, k: int = 3) -> list:
    """
    예측 근거 패턴(공통 IPC·인용·출원인)이 일치하는 훈련 소송 사례 반환.
    - 쿼리 쌍이 가진 공통 특징과 동일한 패턴을 가지면서 실제 소송으로 간 훈련 사례를 찾음.
    - 패턴 매칭 점수가 0이면 임베딩 유사도 fallback.
    """
    pl      = state["pat_list"]
    ckpt    = state["ckpt"]
    feat    = _precompute_pat_features(state)
    ipc_map = feat["ipc"]; cit_map = feat["cit"]; asgn_map = feat["asgn"]

    tp = str(pl[test_idx]); dp = str(pl[target_idx])

    # ── 쿼리 쌍의 공통 패턴 ─────────────────────────────────
    shared_ipc    = ipc_map.get(tp, set())  & ipc_map.get(dp, set())
    shared_cit    = cit_map.get(tp, set())  & cit_map.get(dp, set())
    shared_co     = asgn_map.get(tp, set()) & asgn_map.get(dp, set())
    has_direct    = (dp in cit_map.get(tp, set())) or (tp in cit_map.get(dp, set()))

    # 공통 패턴이 아무것도 없으면 embedding fallback
    if not shared_ipc and not shared_cit and not shared_co and not has_direct:
        return _similar_train_cases(state, test_idx, target_idx, k)

    # ── 훈련 소송 쌍 필터 ────────────────────────────────────
    sued_e  = np.array(ckpt["sued_edges"])
    sued_c  = ckpt["sued_cases"]
    test_cs = set(ckpt["test_cases"])
    tv_mask = [c not in test_cs for c in sued_c]
    tv_s    = [int(sued_e[0, i]) for i, m in enumerate(tv_mask) if m]
    tv_d    = [int(sued_e[1, i]) for i, m in enumerate(tv_mask) if m]
    tv_c    = [sued_c[i]         for i, m in enumerate(tv_mask) if m]

    results = []
    for ts, td, tc in zip(tv_s, tv_d, tv_c):
        pp = str(pl[ts]); dp2 = str(pl[td])

        t_shared_ipc = ipc_map.get(pp, set())  & ipc_map.get(dp2, set())
        t_shared_cit = cit_map.get(pp, set())  & cit_map.get(dp2, set())
        t_shared_co  = asgn_map.get(pp, set()) & asgn_map.get(dp2, set())
        t_has_direct = (dp2 in cit_map.get(pp, set())) or (pp in cit_map.get(dp2, set()))

        score = 0.0
        reasons = []

        # 공통 IPC: 쿼리 쌍이 공유하는 IPC 중 훈련 쌍도 공유하는 비율
        ipc_match = shared_ipc & t_shared_ipc
        if ipc_match:
            jacc = len(ipc_match) / len(shared_ipc | t_shared_ipc)
            score += 3.0 * jacc
            reasons.append(f"공통 IPC {len(ipc_match)}개 일치 ({', '.join(sorted(ipc_match)[:3])})")

        # 직접 인용
        if has_direct and t_has_direct:
            score += 2.5
            reasons.append("직접 인용 관계")

        # 공동 선행기술
        cit_match = shared_cit & t_shared_cit
        if cit_match:
            score += min(len(cit_match) * 0.4, 1.5)
            reasons.append(f"공동 선행기술 {len(cit_match)}건")

        # 동일 출원인
        co_match = shared_co & t_shared_co
        if co_match:
            score += 2.5
            reasons.append(f"동일 출원인: {', '.join(sorted(co_match))}")

        if score > 0:
            results.append({"case": tc, "P": pp, "D": dp2,
                             "score": score, "reasons": reasons})

    results.sort(key=lambda x: -x["score"])
    # 결과가 k개 미만이면 나머지를 embedding fallback으로 채움
    if len(results) < k:
        fb = _similar_train_cases(state, test_idx, target_idx, k)
        seen = {r["case"] for r in results}
        for f in fb:
            if f["case"] not in seen:
                results.append(f)
            if len(results) >= k:
                break
    return results[:k]


# ══════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════
state          = _load()
pat_list       = state["pat_list"]
best_threshold = state["best_threshold"]
tv_plt_set     = state["tv_plt_set"]
tv_def_set     = state["tv_def_set"]
test_pat_set   = state["test_pat_set"]

if "sel_idx"   not in st.session_state: st.session_state.sel_idx   = None
if "pred_mode" not in st.session_state: st.session_state.pred_mode = "threshold"
if "pred_thr"  not in st.session_state: st.session_state.pred_thr  = best_threshold
if "pred_k"    not in st.session_state: st.session_state.pred_k    = 20
if "xai_pair"  not in st.session_state: st.session_state.xai_pair  = None   # (src, dst)
if "xai_run"   not in st.session_state: st.session_state.xai_run   = {}     # {pair_key: {m1:.., m5:.., m6:.., m7:..}}

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='color:#00FFFF;text-shadow:0 0 18px #00FFFF99;"
        "font-size:1.35rem;margin-bottom:2px'>⚖️ PatentNet Galaxy</h2>"
        "<p style='color:#444;font-size:.72rem;margin:0 0 10px'>"
        "GAT + Margin (phase3)  |  2-Hop Injection</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    mode = option_menu(
        menu_title=None,
        options=["🌌 Galaxy", "📊 Statistics"],
        icons=["stars", "bar-chart-fill"],
        default_index=0,
        styles={
            "container":         {"background-color": "transparent", "padding": "0"},
            "icon":              {"color": "#00FFFF", "font-size": "13px"},
            "nav-link":          {"color": "#888", "font-size": "13px",
                                  "border-radius": "8px", "margin": "2px 0"},
            "nav-link-selected": {"background-color": "rgba(0,255,255,0.1)",
                                  "color": "#00FFFF", "font-weight": "700",
                                  "border": "1px solid rgba(0,255,255,0.25)"},
        },
    )
    st.divider()

    if mode == "🌌 Galaxy":
        st.markdown(
            f"<div style='color:#444;font-size:.68rem;margin-bottom:6px'>"
            f"threshold: <span style='color:#00FFFF'>{best_threshold:.2f}</span> "
            f"(val F1 최대)</div>",
            unsafe_allow_html=True,
        )

        sel = st.session_state.sel_idx
        if sel is not None:
            col_r, col_s = st.columns([1, 2])
            with col_r:
                if st.button("✕ 초기화", use_container_width=True):
                    st.session_state.sel_idx = None
                    st.rerun()
            with col_s:
                st.markdown(
                    f"<div style='color:#00FFFF;font-size:.78rem;padding:6px 0'>"
                    f"선택됨: <b>{str(pat_list[sel])[:18]}</b></div>",
                    unsafe_allow_html=True,
                )
            st.divider()

        search = st.text_input("", placeholder="특허 번호 검색...",
                               label_visibility="collapsed")

        # ── 예측 모드 컨트롤 ──────────────────────────────
        st.divider()
        st.markdown("<div style='color:#888;font-size:.7rem;letter-spacing:.08em;"
                    "text-transform:uppercase;margin-bottom:4px'>예측 모드</div>",
                    unsafe_allow_html=True)
        _mode_idx = 0 if st.session_state.pred_mode == "threshold" else 1
        pred_mode_sel = st.radio("", ["🎯 Threshold", "🔢 Top-K"],
                                 index=_mode_idx, horizontal=True,
                                 label_visibility="collapsed")
        if pred_mode_sel == "🎯 Threshold":
            st.session_state.pred_mode = "threshold"
            st.session_state.pred_thr  = st.slider(
                "Threshold", 0.30, 0.95, st.session_state.pred_thr, 0.05,
                help=f"Val F1 최대 기준값: {best_threshold:.2f}")
        else:
            st.session_state.pred_mode = "topk"
            st.session_state.pred_k    = st.slider(
                "Top-K", 1, 100, st.session_state.pred_k,
                help="확률 상위 K개 반환")
        st.divider()

        tab_test, tab_tv = st.tabs([
            f"🧪 대기실 ({len(test_pat_set)})",
            f"📋 Train/Val ({len(tv_plt_set | tv_def_set)})",
        ])

        def _render_pat_list(indices, key_prefix):
            for idx in indices[:60]:
                pid    = str(pat_list[idx])
                is_plt = idx in tv_plt_set
                is_def = idx in tv_def_set
                is_tst = idx in test_pat_set
                is_sel = idx == st.session_state.sel_idx

                role_html = ""
                if is_plt: role_html += "<span class='role-p'>P</span>"
                if is_def: role_html += "<span class='role-d'>D</span>"
                if is_tst: role_html += "<span style='background:#FF0066;color:#FFF;font-size:.6rem;padding:1px 4px;border-radius:3px;margin-left:2px'>TEST</span>"

                border = "#00FFFF" if is_sel else "rgba(0,255,255,0.12)"
                bg     = "rgba(0,255,255,0.10)" if is_sel else "rgba(0,255,255,0.03)"

                st.markdown(
                    f"<div class='wait-item' style='border-color:{border};background:{bg}'>"
                    f"<div class='pid-tag'>{pid}</div>"
                    f"<div style='font-size:.68rem;color:#555;margin-top:3px'>"
                    f"{role_html}</div></div>",
                    unsafe_allow_html=True,
                )
                if is_sel:
                    st.button("✓ 선택됨", key=f"{key_prefix}_{idx}",
                              use_container_width=True, disabled=True)
                else:
                    if st.button("🛸 선택", key=f"{key_prefix}_{idx}",
                                 use_container_width=True):
                        st.session_state.sel_idx = idx
                        st.rerun()
            if len(indices) > 60:
                st.caption(f"… {len(indices)-60}개 더 있음 (검색으로 필터)")

        with tab_test:
            test_sorted = sorted(test_pat_set)
            if search:
                test_sorted = [i for i in test_sorted
                               if search.lower() in str(pat_list[i]).lower()]
            _render_pat_list(test_sorted, "t")

        with tab_tv:
            def _sort_key(i):
                if i in tv_plt_set: return 0
                if i in tv_def_set: return 1
                return 2
            tv_sorted = sorted(tv_plt_set | tv_def_set, key=_sort_key)
            if search:
                tv_sorted = [i for i in tv_sorted
                             if search.lower() in str(pat_list[i]).lower()]
            _render_pat_list(tv_sorted, "v")

        # ── GT 소송 상대 목록 (XAI 연동) ─────────────────────
        if st.session_state.sel_idx is not None:
            _s = st.session_state.sel_idx
            _te = state["test_edges"]; _te_arr = np.array(_te)
            _gt_sb = (set(_te_arr[1, _te_arr[0] == _s].tolist()) |
                      set(_te_arr[0, _te_arr[1] == _s].tolist()))
            if _gt_sb:
                st.divider()
                st.markdown(
                    "<div style='color:#00FF7F;font-size:.7rem;font-weight:700;"
                    "letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px'>"
                    "⚖️ GT 소송 상대 → XAI 분석</div>",
                    unsafe_allow_html=True)
                _cur_thr = st.session_state.pred_thr
                for _gi in sorted(_gt_sb):
                    _gp = 1 / (1 + np.exp(-float(np.dot(state["h_emb"][_s], state["h_emb"][_gi]))))
                    _is_tp = _gp >= _cur_thr
                    _active = st.session_state.xai_pair == (_s, _gi)
                    _badge = "✓ TP" if _is_tp else "✗ FN"
                    _bc = "#00C853" if _is_tp else "#FF6B35"
                    _pid = str(pat_list[_gi])
                    st.markdown(
                        f"<div style='background:rgba(0,255,127,0.05);"
                        f"border:1px solid {'#00FF7F' if _active else 'rgba(0,255,127,0.15)'};"
                        f"border-radius:8px;padding:6px 10px;margin-bottom:4px'>"
                        f"<span style='color:{_bc};font-size:.65rem;font-weight:700'>{_badge}</span>"
                        f"<span style='color:#CCC;font-size:.72rem;margin-left:6px'>{_pid}</span><br>"
                        f"<span style='color:#555;font-size:.65rem'>P={_gp:.3f}</span>"
                        f"</div>", unsafe_allow_html=True)
                    if st.button("🧠 XAI", key=f"xai_sb_{_gi}", use_container_width=True):
                        st.session_state.xai_pair = (_s, _gi)
                        st.rerun()

        st.divider()
        st.markdown("""
<div style='font-size:.72rem;color:#444;line-height:2'>
<b style='color:#666'>Galaxy 노드</b><br>
⬤ <span style='color:#FFD700'>Gold</span> 원고특허 (train/val)<br>
⬤ <span style='color:#FF6B35'>Orange</span> 피고특허 (train/val)<br>
⬤ <span style='color:#007B9E'>Blue</span> 기타 특허<br>
◆ <span style='color:#32CD32'>Green</span> IPC<br>
■ <span style='color:#FF00FF'>Magenta</span> 회사<br>
⬤ <span style='color:#FFFFFF'>White★</span> 선택 특허<br><br>
<b style='color:#666'>예측 엣지 (확률)</b><br>
━ <span style='color:#FF0066'>Red</span>    ≥ 0.80<br>
━ <span style='color:#FF6600'>Orange</span> ≥ 0.65<br>
━ <span style='color:#FFAA00'>Yellow</span> ≥ threshold<br>
✅ 없음: threshold 미만
</div>""", unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────────
sel_idx = st.session_state.sel_idx

if mode == "🌌 Galaxy":

    # ── A: 기본 Galaxy (캐시된 HTML 재사용) ──────────────
    if sel_idx is None:
        colored_header(
            label="소송 특허 생태계 — Galaxy View",
            description=(f"Train+Val {len(state['galaxy_pat_set'])}개 특허  "
                         f"| 대기실에서 Test 특허를 클릭해 주입하세요"),
            color_name="blue-70",
        )
        # ② HTML은 한 번만 만들어 cache_data로 고정 → 재렌더 0초
        base_html = _build_base_galaxy_html(0)
        components.html(base_html, height=840, scrolling=False)

    # ── B: 예측 엣지 주입 뷰 ─────────────────────────────
    else:
        test_pid = str(pat_list[sel_idx])

        # 예측 모드 분기
        _thr = st.session_state.pred_thr
        _k   = st.session_state.pred_k
        if st.session_state.pred_mode == "threshold":
            preds = predict_edges(state["h_emb"], sel_idx, threshold=_thr)
            preds_label = f"Threshold={_thr:.2f}"
        else:
            preds = predict_top_k(state["h_emb"], sel_idx, k=_k)
            preds_label = f"Top-{_k}"

        # 비교용: 항상 두 가지 모두 계산
        preds_thr  = predict_edges(state["h_emb"], sel_idx, threshold=_thr)
        preds_topk = predict_top_k(state["h_emb"], sel_idx, k=_k)

        # ground truth
        te     = state["test_edges"]
        te_arr = np.array(te)
        gt_set = (set(te_arr[1, te_arr[0] == sel_idx].tolist()) |
                  set(te_arr[0, te_arr[1] == sel_idx].tolist()))

        # 예측 없을 경우 fallback: top-5
        no_pred = len(preds) == 0
        if no_pred:
            preds = predict_top_k(state["h_emb"], sel_idx, k=5)

        best_prob = preds[0][1] if preds else 0.0
        is_plt    = sel_idx in tv_plt_set
        is_def    = sel_idx in tv_def_set
        role_str  = ("P/D" if (is_plt and is_def) else
                     "P"   if is_plt else
                     "D"   if is_def else "—")
        top_lbl, top_col, top_pct = _risk(best_prob)

        colored_header(
            label=f"🛸 {test_pid} — 소송 예측",
            description=(f"모드: {preds_label}  |  예측 {len(preds)}개  |  GT {len(gt_set)}개"
                         + ("  [fallback: top-5]" if no_pred else "")),
            color_name="blue-70",
        )

        # ── 상단 메트릭 카드 ──────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            f'<div class="glass-card"><div class="metric-label">선택 특허</div>'
            f'<div class="metric-cyan" style="font-size:1.05rem">{test_pid}</div></div>',
            unsafe_allow_html=True)
        c2.markdown(
            f'<div class="glass-card"><div class="metric-label">소송 역할 (train/val)</div>'
            f'<div class="metric-gold">{role_str}</div></div>',
            unsafe_allow_html=True)
        c3.markdown(
            f'<div class="glass-card"><div class="metric-label">예측 엣지 수</div>'
            f'<div class="metric-cyan">{len(preds)}개</div></div>',
            unsafe_allow_html=True)
        c4.markdown(
            f'<div class="glass-card"><div class="metric-label">Top-1 확률</div>'
            f'<div style="font-size:1.8rem;font-weight:700;color:{top_col};'
            f'text-shadow:0 0 14px {top_col}88">{best_prob:.2%}</div></div>',
            unsafe_allow_html=True)
        style_metric_cards(background_color="#050810", border_color="#00FFFF18")

        # ── 예측 엣지 없음 분기 ──────────────────────────
        if no_pred:
            st.markdown(
                f"<div style='margin:20px auto;text-align:center;max-width:600px'>"
                f"<div style='font-size:2.5rem'>✅</div>"
                f"<div style='font-size:1.3rem;font-weight:700;color:#32CD32;"
                f"text-shadow:0 0 20px #32CD3288;margin:12px 0'>소송 위험 없음</div>"
                f"<div style='color:#555;font-size:.9rem'>"
                f"threshold {best_threshold:.2f} 이상의 예측 엣지 없음<br>"
                f"(아래는 참고용 top-5)</div>"
                f"</div>",
                unsafe_allow_html=True)
        else:
            # ── 전체 Galaxy + 2-hop + t-SNE 탭 ──────────────
            tab_full, tab_sub, tab_tsne, tab_xai = st.tabs([
                "🌌 전체 Galaxy 주입", "🔍 2-Hop 서브그래프",
                "🔬 t-SNE 임베딩", "🧠 XAI 분석"])

            with tab_full:
                st.caption("흰색 큰 노드=Test 특허  |  초록 엣지/테두리=Ground Truth  |  네온 엣지=예측 소송 위험  |  빨간 엣지=기존 소송")
                full_html = _build_full_injection_html(state, sel_idx, preds, gt_set=gt_set)
                components.html(full_html, height=860, scrolling=False)

            with tab_sub:
                n_hops = st.slider(
                    "탐색 범위 (hop)", min_value=1, max_value=4, value=2,
                    help="hop이 클수록 더 넓은 범위의 특허가 표시됩니다 (citation + 소송 엣지 모두 탐색)",
                    label_visibility="visible",
                )
                st.caption(
                    f"Test 노드 중심 {n_hops}-hop 이내 특허 표시 "
                    f"(인용·소송 엣지 기준) — IPC·회사 노드 포함"
                )
                inj_html = _build_injection_html(state, sel_idx, preds, n_hops=n_hops, gt_set=gt_set)
                components.html(inj_html, height=800, scrolling=False)

            with tab_tsne:
                t_col1, t_col2 = st.columns([5, 1])
                with t_col2:
                    perp = st.slider("Perplexity", 5, 80, 30,
                                     help="t-SNE perplexity. 값 바꾸면 재계산 (캐시됨)")
                    st.markdown(
                        "<div style='font-size:.68rem;color:#555;margin-top:16px;line-height:2'>"
                        "<b style='color:#888'>범례</b><br>"
                        "★ 흰별 = 선택특허<br>"
                        "◆ 주황+흰테두리 = GT TP<br>"
                        "◆ 주황 = GT FN<br>"
                        "● 색상 = P(sue)<br>"
                        "● 크기 = 소송횟수"
                        "</div>", unsafe_allow_html=True)
                with t_col1:
                    xy = _tsne_xy(state["h_emb"], perplexity=perp)
                    fig = _build_tsne_fig(
                        xy=xy,
                        h_emb=state["h_emb"],
                        pat_list=pat_list,
                        sel_idx=sel_idx,
                        gt_set=gt_set,
                        sued_edges=np.array(state["ckpt"]["sued_edges"]),
                        thr=_thr,
                    )
                    tsne_event = st.plotly_chart(
                        fig, use_container_width=True,
                        key="tsne_chart", on_select="rerun",
                        selection_mode="points")
                    # ── GT 다이아 클릭 → xai_pair 설정 ──────────
                    if tsne_event and getattr(tsne_event, 'selection', None):
                        pts = getattr(tsne_event.selection, 'points', [])
                        for pt in pts:
                            cd = getattr(pt, 'customdata', None)
                            if cd is not None:
                                clicked_idx = int(cd[0])
                                if clicked_idx in gt_set:
                                    st.session_state.xai_pair = (sel_idx, clicked_idx)
                                    st.session_state.xai_run  = {}
                                    st.toast(f"XAI 쌍 설정: {pat_list[sel_idx]} ↔ {pat_list[clicked_idx]}", icon="🧠")
                                    st.rerun()

            with tab_xai:
                _xp = st.session_state.xai_pair
                # ── 쌍 미선택 안내 ────────────────────────────
                if _xp is None or _xp[0] != sel_idx:
                    st.info(
                        "**GT 소송 상대를 선택하면 XAI 분석이 시작됩니다.**\n\n"
                        "- 왼쪽 사이드바의 **⚖️ GT 소송 상대** 목록에서 🧠 XAI 버튼 클릭\n"
                        "- 또는 **t-SNE 탭**에서 ◆ 주황 다이아몬드 클릭")
                else:
                    _xs, _xd = _xp
                    _xprob = 1 / (1 + np.exp(-float(np.dot(state["h_emb"][_xs], state["h_emb"][_xd]))))
                    _is_tp = _xprob >= _thr
                    st.markdown(
                        f"<div style='background:rgba(0,255,127,0.06);border:1px solid #00FF7F44;"
                        f"border-radius:10px;padding:12px 16px;margin-bottom:12px'>"
                        f"<span style='color:#00FFFF;font-weight:700'>{pat_list[_xs]}</span>"
                        f"<span style='color:#555;margin:0 10px'>↔</span>"
                        f"<span style='color:#FF8C00;font-weight:700'>{pat_list[_xd]}</span>"
                        f"&nbsp;&nbsp;<span style='color:{'#00C853' if _is_tp else '#FF6B35'};"
                        f"font-size:.8rem'>{'✓ TP' if _is_tp else '✗ FN'}&nbsp;"
                        f"P(sue)={_xprob:.3f}</span></div>",
                        unsafe_allow_html=True)

                    _pair_key = f"{_xs}_{_xd}"
                    _xrc = st.session_state.xai_run.setdefault(_pair_key, {})

                    # ── M1: Attention ─────────────────────────────
                    st.markdown(
                        "<div style='color:#4fc3f7;font-size:1.05rem;font-weight:700;"
                        "margin:12px 0 4px'>M1. Attention Visualization in GAT</div>"
                        "<div style='color:#666;font-size:.8rem;border-left:3px solid #4fc3f7;"
                        "padding-left:8px;margin-bottom:10px'>각 GAT 레이어 × edge type별 attention weight<br>"
                        "→ 어떤 연결 관계가 임베딩 형성에 집중됐는지</div>",
                        unsafe_allow_html=True)
                    if 'm1' not in _xrc:
                        with st.spinner("Attention 계산 중…"):
                            _xrc['m1'] = _compute_attention(_xs, _xd, state)
                    _attn = _xrc['m1']
                    # 고정 색상 매핑 (전 subplot 공통)
                    _REL_COLOR = {
                        'sued':        '#EF5350',
                        'sued_by':     '#FF7043',
                        'cites':       '#42A5F5',
                        'cited_by':    '#26C6DA',
                        'has_ipc':     '#66BB6A',
                        'rev_has_ipc': '#9CCC65',
                        'owns':        '#AB47BC',
                        'owned_by':    '#CE93D8',
                    }
                    _REL_KO = {
                        'sued':        'sued — 소송',
                        'sued_by':     'sued_by — 피소',
                        'cites':       'cites — 인용',
                        'cited_by':    'cited_by — 피인용',
                        'has_ipc':     'has_ipc — IPC분류',
                        'rev_has_ipc': 'rev_has_ipc — IPC(역)',
                        'owns':        'owns — 출원인소유',
                        'owned_by':    'owned_by — 출원인피소유',
                    }
                    _fig_m1, _axes = plt.subplots(2, 3, figsize=(18, 8), facecolor='#050810')
                    _fig_m1.suptitle(f'Attention  {pat_list[_xs]} ↔ {pat_list[_xd]}',
                                     fontsize=11, color='white')
                    _all_used = set()
                    for _ri, (_ni, _nl) in enumerate([
                        (_xs, f'[A] {pat_list[_xs]}'), (_xd, f'[B] {pat_list[_xd]}')]):
                        for _ci, _layer in enumerate(['L1', 'L2', 'L3']):
                            _ax = _axes[_ri][_ci]; _ax.set_facecolor('#0e1117')
                            _ax.tick_params(colors='white', labelsize=7)
                            for sp in _ax.spines.values(): sp.set_edgecolor('#333')
                            _bars = []
                            for _et, _v in _attn.get(_layer, {}).get(_ni, {}).items():
                                _st, _rel, _dt = _et
                                for _sn, _av in zip(_v['src'], _v['attn']):
                                    _bars.append((_node_label(_sn, _st, state)[:14], _av, _rel))
                            if not _bars:
                                _ax.text(0.5, 0.5, 'No edges', ha='center', va='center', color='#555')
                                _ax.axis('off'); continue
                            _bars.sort(key=lambda x: -x[1]); _bars = _bars[:8]
                            _all_used.update(b[2] for b in _bars)
                            _ax.barh(range(len(_bars)), [b[1] for b in _bars],
                                     color=[_REL_COLOR.get(b[2], '#888') for b in _bars],
                                     alpha=0.88)
                            _ax.set_yticks(range(len(_bars)))
                            _ax.set_yticklabels([b[0] for b in _bars], color='white', fontsize=6)
                            _ax.set_title(f'{_nl[:20]} | {_layer}', fontsize=8, color='white')
                            _ax.set_xlabel('Attention weight', color='gray', fontsize=7)
                    # 공통 범례 — figure 하단에 한 번만
                    _legend_handles = [
                        Patch(color=_REL_COLOR[r], label=_REL_KO[r])
                        for r in _REL_COLOR if r in _all_used
                    ]
                    _fig_m1.legend(handles=_legend_handles, loc='lower center',
                                   ncol=len(_legend_handles), fontsize=8,
                                   facecolor='#0e1117', labelcolor='white',
                                   framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
                    plt.tight_layout()
                    plt.subplots_adjust(bottom=0.10)
                    st.pyplot(_fig_m1, use_container_width=True); plt.close(_fig_m1)

                    st.divider()

                    # ── M5: PGExplainer ───────────────────────────
                    st.markdown(
                        "<div style='color:#4fc3f7;font-size:1.05rem;font-weight:700;"
                        "margin:12px 0 4px'>M5. PGExplainer</div>"
                        "<div style='color:#666;font-size:.8rem;border-left:3px solid #4fc3f7;"
                        "padding-left:8px;margin-bottom:10px'>MLP(h_i ‖ h_j) → [0,1] 이웃 노드 중요도<br>"
                        "→ 어떤 주변 특허가 이 소송 예측을 이끌었는지 식별</div>",
                        unsafe_allow_html=True)
                    if 'm5' not in _xrc:
                        if st.button("▶ M5 실행 (~20초)", key=f"m5_{_pair_key}"):
                            with st.spinner("PGExplainer 학습 중…"):
                                _xrc['m5'] = _compute_pg(_xs, _xd, state)
                            st.rerun()
                    else:
                        _sc_src, _sc_dst = _xrc['m5']
                        _cm1, _cm2 = st.columns(2)
                        for _col, _sc, _role, _bc, _lbl in [
                            (_cm1, _sc_src, 'A', '#42a5f5', f'[A] {pat_list[_xs]}'),
                            (_cm2, _sc_dst, 'B', '#ef5350', f'[B] {pat_list[_xd]}')]:
                            with _col:
                                _top = sorted(_sc.items(), key=lambda x: -x[1])[:10]
                                _fig5, _ax5 = plt.subplots(figsize=(7, 5), facecolor='#050810')
                                _ax5.set_facecolor('#0e1117')
                                _ax5.tick_params(colors='white', labelsize=8)
                                for sp in _ax5.spines.values(): sp.set_edgecolor('#333')
                                _ax5.barh(range(len(_top)), [v for _, v in _top],
                                          color=_bc, alpha=0.8)
                                _ax5.set_yticks(range(len(_top)))
                                _ax5.set_yticklabels(
                                    [str(pat_list[n])[:18] for n, _ in _top],
                                    color='white', fontsize=7)
                                _ax5.set_title(f'[{_role}] Neighbor Importance', color='white', fontsize=9)
                                _ax5.set_xlabel('PG Score', color='gray', fontsize=7)
                                plt.tight_layout(); st.pyplot(_fig5, use_container_width=True); plt.close(_fig5)

                    st.divider()

                    # ── M6: SubgraphX ─────────────────────────────
                    st.markdown(
                        "<div style='color:#4fc3f7;font-size:1.05rem;font-weight:700;"
                        "margin:12px 0 4px'>M6. SubgraphX (Monte Carlo Shapley)</div>"
                        "<div style='color:#666;font-size:.8rem;border-left:3px solid #4fc3f7;"
                        "padding-left:8px;margin-bottom:10px'>이웃 노드의 Shapley value (Monte Carlo 추정)<br>"
                        "→ 양수=예측에 기여 / 음수=예측을 방해하는 노드</div>",
                        unsafe_allow_html=True)
                    if 'm6' not in _xrc:
                        if st.button("▶ M6 실행 (~2분)", key=f"m6_{_pair_key}"):
                            with st.spinner("SubgraphX Shapley 계산 중… (~2분)"):
                                _xrc['m6'] = _compute_subgraphx(_xs, _xd, state)
                            st.rerun()
                    else:
                        _sv = _xrc['m6']
                        _sv_sorted = sorted(_sv.items(), key=lambda x: -abs(x[1]))[:15]
                        _fig6, _ax6 = plt.subplots(figsize=(10, 6), facecolor='#050810')
                        _ax6.set_facecolor('#0e1117')
                        _ax6.tick_params(colors='white', labelsize=9)
                        for sp in _ax6.spines.values(): sp.set_edgecolor('#333')
                        _ax6.barh(range(len(_sv_sorted)),
                                  [v for _, v in _sv_sorted],
                                  color=['#ef5350' if v > 0 else '#42a5f5' for _, v in _sv_sorted],
                                  alpha=0.85)
                        _ax6.axvline(0, color='white', linewidth=1)
                        _ax6.set_yticks(range(len(_sv_sorted)))
                        _ax6.set_yticklabels([str(pat_list[n])[:20] for n, _ in _sv_sorted],
                                             color='white', fontsize=9)
                        _ax6.set_xlabel('Shapley Value  (+: helpful,  −: harmful)',
                                        color='gray', fontsize=9)
                        _ax6.set_title('Shapley Value by Neighbor Node', color='white', fontsize=11)
                        _ax6.legend(handles=[Patch(color='#ef5350', label='Helpful (+)'),
                                             Patch(color='#42a5f5', label='Harmful (−)')],
                                    facecolor='#0e1117', labelcolor='white')
                        plt.tight_layout(); st.pyplot(_fig6, use_container_width=True); plt.close(_fig6)

                    st.divider()

                    # ── M7: GraphSHAP ─────────────────────────────
                    st.markdown(
                        "<div style='color:#4fc3f7;font-size:1.05rem;font-weight:700;"
                        "margin:12px 0 4px'>M7. GraphSHAP (KernelSHAP Feature Attribution)</div>"
                        "<div style='color:#666;font-size:.8rem;border-left:3px solid #4fc3f7;"
                        "padding-left:8px;margin-bottom:10px'>입력 피처를 그룹(struct 3d / FAISS 구간)으로 나눠 Shapley 추정<br>"
                        "→ FAISS 임베딩 vs 그래프 구조 기여도 정량화</div>",
                        unsafe_allow_html=True)
                    if 'm7' not in _xrc:
                        if st.button("▶ M7 실행 (~1분)", key=f"m7_{_pair_key}"):
                            with st.spinner("GraphSHAP KernelSHAP 계산 중… (~1분)"):
                                _xrc['m7'] = _compute_graphshap(_xs, _xd, state)
                            st.rerun()
                    else:
                        _sh_src, _sh_dst = _xrc['m7']
                        _cm3, _cm4 = st.columns(2)
                        for _col, _sv7, _role, _lbl in [
                            (_cm3, _sh_src, 'A', pat_list[_xs]),
                            (_cm4, _sh_dst, 'B', pat_list[_xd])]:
                            with _col:
                                _slist = sorted(_sv7.items(), key=lambda x: -abs(x[1]))
                                _fig7, _ax7 = plt.subplots(figsize=(8, 6), facecolor='#050810')
                                _ax7.set_facecolor('#0e1117')
                                _ax7.tick_params(colors='white', labelsize=8)
                                for sp in _ax7.spines.values(): sp.set_edgecolor('#333')
                                _ax7.barh(range(len(_slist)), [v for _, v in _slist],
                                          color=['#4caf50' if v > 0 else '#ef5350' for _, v in _slist],
                                          alpha=0.85)
                                _ax7.axvline(0, color='white', linewidth=1)
                                _ax7.set_yticks(range(len(_slist)))
                                _ax7.set_yticklabels([g for g, _ in _slist], color='white', fontsize=8)
                                _ax7.set_title(f'[{_role}] {str(_lbl)[:18]}\nFeature Group SHAP',
                                               color='white', fontsize=9)
                                _ax7.set_xlabel('SHAP Value (contribution)', color='gray', fontsize=8)
                                plt.tight_layout(); st.pyplot(_fig7, use_container_width=True); plt.close(_fig7)

            # ── 성능 비교 패널 ────────────────────────────────
            st.divider()
            colored_header(label="📊 예측 모드별 성능 비교",
                           description=f"GT {len(gt_set)}개 기준  |  이 특허에 대한 단건 성능",
                           color_name="orange-70")

            p_thr   = _perf(preds_thr,  gt_set)
            p_topk  = _perf(preds_topk, gt_set)

            cm1, cm2 = st.columns(2)
            def _perf_card(col, label, p, active):
                border = "border:1px solid #00FFFF44;" if active else ""
                col.markdown(
                    f"<div class='glass-card' style='padding:14px 16px;{border}'>"
                    f"<div style='color:#00FFFF;font-size:.78rem;font-weight:700;"
                    f"margin-bottom:8px'>{label}"
                    + (" <span style='background:#00FFFF22;padding:1px 6px;border-radius:3px;"
                       "font-size:.65rem'>현재</span>" if active else "") +
                    f"</div>"
                    f"<table style='width:100%;font-size:.75rem;border-collapse:collapse'>"
                    f"<tr><td style='color:#888'>예측 수</td>"
                    f"<td style='color:#FFF;text-align:right;font-weight:600'>{p['n_pred']}개</td></tr>"
                    f"<tr><td style='color:#888'>GT 적중</td>"
                    f"<td style='color:#00FF7F;text-align:right;font-weight:600'>{p['hits']} / {p['n_gt']}</td></tr>"
                    f"<tr><td style='color:#888'>Precision</td>"
                    f"<td style='color:#FFD700;text-align:right;font-weight:700'>{p['prec']:.3f}</td></tr>"
                    f"<tr><td style='color:#888'>Recall</td>"
                    f"<td style='color:#FFD700;text-align:right;font-weight:700'>{p['rec']:.3f}</td></tr>"
                    f"<tr><td style='color:#888'>F1</td>"
                    f"<td style='color:#FF6600;text-align:right;font-weight:700'>{p['f1']:.3f}</td></tr>"
                    f"</table></div>",
                    unsafe_allow_html=True)

            _perf_card(cm1, f"🎯 Threshold={_thr:.2f}",
                       p_thr,  st.session_state.pred_mode == "threshold")
            _perf_card(cm2, f"🔢 Top-{_k}",
                       p_topk, st.session_state.pred_mode == "topk")

            # ── 전체 test 집계 성능 ───────────────────────────
            with st.expander("📈 전체 Test set 집계 성능 보기"):
                agg = _agg_perf(state["h_emb"], te_arr, _thr, _k)
                a_t, a_k = agg["threshold"], agg["topk"]
                st.markdown(
                    f"<table style='width:100%;font-size:.78rem;border-collapse:collapse;"
                    f"color:#CCC'>"
                    f"<thead><tr style='border-bottom:1px solid #333'>"
                    f"<th style='text-align:left;padding:6px 8px'>지표</th>"
                    f"<th style='text-align:right;padding:6px 8px;color:#00FFFF'>"
                    f"Threshold={_thr:.2f}</th>"
                    f"<th style='text-align:right;padding:6px 8px;color:#FF6600'>"
                    f"Top-{_k}</th></tr></thead><tbody>"
                    + "".join(
                        f"<tr style='border-bottom:1px solid #1a1a1a'>"
                        f"<td style='padding:5px 8px;color:#888'>{lbl}</td>"
                        f"<td style='padding:5px 8px;text-align:right;color:#00FFFF;font-weight:600'>{vt}</td>"
                        f"<td style='padding:5px 8px;text-align:right;color:#FF6600;font-weight:600'>{vk}</td></tr>"
                        for lbl, vt, vk in [
                            ("대상 특허 수", a_t["n_pats"], a_k["n_pats"]),
                            ("총 예측 수",   a_t["n_pred"], a_k["n_pred"]),
                            ("총 GT 수",     a_t["n_gt"],   a_k["n_gt"]),
                            ("총 적중 수",   a_t["hits"],   a_k["hits"]),
                            ("Macro Prec",   f"{a_t['prec']:.4f}", f"{a_k['prec']:.4f}"),
                            ("Macro Rec",    f"{a_t['rec']:.4f}",  f"{a_k['rec']:.4f}"),
                            ("Macro F1",     f"{a_t['f1']:.4f}",   f"{a_k['f1']:.4f}"),
                        ])
                    + "</tbody></table>",
                    unsafe_allow_html=True)

            # ── 우측 패널: Top-10 + 케이스 ───────────────────
            col_g, col_t = st.columns([3, 1])
            with col_t:
                gt_hit = sum(1 for di, _ in preds if di in gt_set)
                colored_header(
                    label=f"예측 소송 엣지 {len(preds)}개",
                    description=(f"{preds_label}"
                                 f"  |  ✓ GT {gt_hit}/{len(gt_set)}개 적중"),
                    color_name="violet-70")
                for rank, (di, prob) in enumerate(preds[:10], 1):
                    lbl, col, pct = _risk(prob)
                    bar = "█" * (int(prob * 10)) + "░" * (10 - int(prob * 10))
                    is_gt = di in gt_set
                    gt_badge = ("<span style='background:#00C853;color:#000;"
                                "font-size:.62rem;font-weight:800;padding:1px 5px;"
                                "border-radius:4px;margin-left:4px'>✓ GT</span>"
                                if is_gt else "")
                    card_border = "border:1px solid #00C853;" if is_gt else ""
                    st.markdown(
                        f"<div class='glass-card' style='padding:10px 14px;margin-bottom:6px;{card_border}'>"
                        f"<div style='display:flex;justify-content:space-between'>"
                        f"<span style='color:#888;font-size:.68rem'>#{rank}</span>"
                        f"<span style='color:{col};font-size:.68rem;font-weight:700'>{lbl}</span>"
                        f"</div>"
                        f"<div style='color:#FFF;font-size:.78rem;font-weight:600;"
                        f"margin:3px 0'>{str(pat_list[di])}{gt_badge}</div>"
                        f"<div style='color:{col};font-size:.68rem;font-family:monospace'>"
                        f"{bar} {prob:.2%}</div>"
                        f"</div>",
                        unsafe_allow_html=True)

            # ── 공통 특징 + 유사 훈련 사례 ───────────────
            st.divider()
            colored_header(label="예측 근거 분석",
                           description="Top 후보별 공통 특징 & 유사 훈련 사례",
                           color_name="violet-70")

            tabs = st.tabs([f"#{r} {str(pat_list[di])[:14]}.."
                            for r, (di, _) in enumerate(preds[:5], 1)])
            for tab, (di, cos) in zip(tabs, preds[:5]):
                with tab:
                    lbl, col, pct = _risk(cos)
                    sf        = _shared_features(state, sel_idx, di)
                    sim_cases = _similar_train_cases(state, sel_idx, di, k=3)

                    # ── 연결 구조 미니 그래프 ─────────────────
                    reason_html = _build_reason_graph_html(state, sel_idx, di, cos)
                    components.html(reason_html, height=320, scrolling=False)

                    fa, fb = st.columns(2)
                    with fa:
                        st.markdown(
                            f"<div class='glass-card'>"
                            f"<div class='metric-label'>공통 특징</div>",
                            unsafe_allow_html=True)
                        if sf["direct"]:
                            st.markdown("🔗 **직접 인용 관계 있음**")
                        if sf["ipc"]:
                            st.markdown(f"📐 공통 IPC: `{'`, `'.join(sorted(sf['ipc']))}`")
                        if sf["cit"]:
                            st.markdown(f"📄 공동 인용 선행기술: **{len(sf['cit'])}건**")
                        if sf["co"]:
                            st.markdown(f"🏢 동일 출원인: `{'`, `'.join(sorted(sf['co']))}`")
                        if not any([sf["direct"], sf["ipc"], sf["cit"], sf["co"]]):
                            st.markdown("_공통 구조적 특징 없음 — 임베딩 기반 유사도_")
                        st.markdown("</div>", unsafe_allow_html=True)

                    with fb:
                        st.markdown(
                            f"<div class='glass-card'>"
                            f"<div class='metric-label'>유사 훈련 소송 사례</div>",
                            unsafe_allow_html=True)
                        for sc in sim_cases:
                            st.markdown(
                                f"<div style='border-left:2px solid #FF0066;"
                                f"padding-left:10px;margin-bottom:10px'>"
                                f"<div style='color:#888;font-size:.65rem'>"
                                f"CASE {sc['case']}  |  유사도 {sc['sim']:.3f}</div>"
                                f"<div style='color:#FFD700;font-size:.72rem'>"
                                f"원고: {sc['P']}</div>"
                                f"<div style='color:#FF6B35;font-size:.72rem'>"
                                f"피고: {sc['D']}</div>"
                                f"</div>",
                                unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

# ── Statistics ────────────────────────────────────────────
else:
    colored_header(label="데이터셋 통계",
                   description="GAT + InfoNCE 학습 기준",
                   color_name="blue-70")

    c1, c2, c3, c4 = st.columns(4)
    for col, (lb, v, color) in zip(
        [c1, c2, c3, c4],
        [("전체 특허",          len(pat_list),                   "#00FFFF"),
         ("Galaxy (Train+Val)", len(state["galaxy_pat_set"]),    "#FFD700"),
         ("대기실 (Test)",       len(state["test_pat_set"]),     "#FF0066"),
         ("소송 케이스",          269,                            "#FF6B35")],
    ):
        col.markdown(
            f'<div class="glass-card"><div class="metric-label">{lb}</div>'
            f'<div style="font-size:2.2rem;font-weight:700;color:{color};'
            f'text-shadow:0 0 10px {color}55">{v:,}</div></div>',
            unsafe_allow_html=True)

    colored_header(label="모델 성능",
                   description="Test set 기준 (27 cases, 165 pairs)",
                   color_name="violet-70")
    st.dataframe(pd.DataFrame({
        "Model":       ["FAISS (baseline)", "SAGE+Margin", "SAGE+InfoNCE",
                        "GAT+InfoNCE", "GAT+Margin (phase3) ★"],
        "AUROC":       [0.9301, 0.9070, 0.8563, 0.9193, 0.9126],
        "AUPRC":       [0.8961, 0.8585, 0.7876, 0.8851, 0.8871],
        "MRR":         [0.1817, 0.1213, 0.0925, 0.1271, 0.1333],
        "Hits@10":     [0.5152, 0.3394, 0.2000, 0.3394, 0.3515],
        "Median Rank": [10,     41,      89,      28,     18],
    }), use_container_width=True, hide_index=True)
    st.caption("★ 현재 Galaxy 대시보드 — GAT+Margin, BatchNorm, Tanh(L3), k=4")

    colored_header(label="고속화 전략",
                   description="지연 없는 실시간 렌더링",
                   color_name="blue-70")
    st.dataframe(pd.DataFrame({
        "전략":    ["① cache_resource", "② 기본 Galaxy HTML 캐시", "③ 2-hop 서브그래프"],
        "효과":    ["모델·임베딩 RAM 상주", "전체 galaxy 재빌드 없음", "클릭 시 ~100노드만 렌더"],
        "지연":    ["최초 2~3초 (이후 0초)", "최초 1회 (이후 0초)", "<1초"],
    }), use_container_width=True, hide_index=True)
