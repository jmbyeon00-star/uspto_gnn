"""demo_app.py — Interactive Patent Litigation GNN Demo.

Flow:
  1. Full 1331-patent Galaxy (spring layout, stable positions)
  2. Sidebar: test plaintiff selector → top-20 predicted edges overlay
  3. Click GT edge (green) → 2-hop hetero focus (non-2-hop → black)
  4. "Run XAI" → compute attention → layer-by-layer animation on same canvas
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Patent Litigation GNN Demo",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import streamlit.components.v1 as components
from pyvis.network import Network

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.loader import load_state
from backend.model import EDGE_TYPES

RESULTS_DIR = Path(__file__).parent.parent.parent / "0713" / "results"

# ── CSS ────────────────────────────────────────────────────────────────────────
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
  .stButton > button {
    background: linear-gradient(135deg,#00FFFF18,#00BFFF18);
    border: 1px solid #00FFFF55; color:#00FFFF;
    border-radius:8px; font-family:'JetBrains Mono',monospace;
    font-weight:700; transition:all .2s; width:100%;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg,#00FFFF33,#00BFFF33);
    border-color:#00FFFF; box-shadow:0 0 14px #00FFFF33;
  }
  .xai-btn > button {
    background: linear-gradient(135deg,#FF006618,#FF440018) !important;
    border: 1px solid #FF006655 !important; color:#FF4444 !important;
  }
  .xai-btn > button:hover {
    background: linear-gradient(135deg,#FF006633,#FF440033) !important;
    border-color:#FF0066 !important; box-shadow:0 0 14px #FF006633 !important;
  }
  hr { border-color:rgba(255,255,255,0.06) !important; }
  ::-webkit-scrollbar { width:5px; }
  ::-webkit-scrollbar-track { background:#050810; }
  ::-webkit-scrollbar-thumb { background:#00FFFF33; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🌌 모델 & 그래프 로딩 중...")
def _load():
    return load_state()

@st.cache_data(show_spinner=False)
def _load_full_graph() -> dict:
    p = RESULTS_DIR / "full_graph.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner="📡 그래프 adjacency 구축 중...")
def _build_hetero_adj(_key: int) -> dict:
    """Build patent-centric adjacency via ALL edge types. Cached."""
    state = _load()
    data  = state["data"]

    pat_pat: dict[int, set] = defaultdict(set)
    pat_co:  dict[int, set] = defaultdict(set)
    co_pat:  dict[int, set] = defaultdict(set)
    pat_ipc: dict[int, set] = defaultdict(set)
    ipc_pat: dict[int, set] = defaultdict(set)

    for et in EDGE_TYPES:
        s_t, rel, d_t = et
        if et not in data.edge_index_dict:
            continue
        ei = data.edge_index_dict[et].cpu().numpy()
        ss, dd = ei[0].tolist(), ei[1].tolist()

        if s_t == "patent" and d_t == "patent":
            for s, d in zip(ss, dd):
                pat_pat[s].add(d); pat_pat[d].add(s)
        elif s_t == "patent" and d_t == "company":
            for s, d in zip(ss, dd): pat_co[s].add(d); co_pat[d].add(s)
        elif s_t == "company" and d_t == "patent":
            for s, d in zip(ss, dd): co_pat[s].add(d); pat_co[d].add(s)
        elif s_t == "patent" and d_t == "ipc":
            for s, d in zip(ss, dd): pat_ipc[s].add(d); ipc_pat[d].add(s)
        elif s_t == "ipc" and d_t == "patent":
            for s, d in zip(ss, dd): ipc_pat[s].add(d); pat_ipc[d].add(s)

    # Convert sets to lists for JSON-serializability
    return {
        "pp": {k: list(v) for k, v in pat_pat.items()},
        "pc": {k: list(v) for k, v in pat_co.items()},
        "cp": {k: list(v) for k, v in co_pat.items()},
        "pi": {k: list(v) for k, v in pat_ipc.items()},
        "ip": {k: list(v) for k, v in ipc_pat.items()},
    }


def _hop2_patents(adj: dict, src: int, dst: int) -> list[int]:
    """2-hop patent set via ALL edge types (patent→company→patent, etc.)."""
    pp = adj["pp"]; pc = adj["pc"]; cp = adj["cp"]
    pi = adj["pi"]; ip = adj["ip"]

    def _expand(anchor: int) -> set[int]:
        hop1_pats = set(pp.get(anchor, []))
        result    = {anchor} | hop1_pats
        for c in pc.get(anchor, []): result.update(cp.get(c, []))   # via company
        for i in pi.get(anchor, []): result.update(ip.get(i, []))   # via ipc
        for p1 in hop1_pats:
            result.update(pp.get(p1, []))
            for c in pc.get(p1, []): result.update(cp.get(c, []))
            for i in pi.get(p1, []): result.update(ip.get(i, []))
        return result

    return list(_expand(src) | _expand(dst))


# ── Attention computation (재사용 from app.py) ─────────────────────────────────
def _compute_attention(src_idx: int, dst_idx: int) -> dict:
    """Per-layer GAT attention for src & dst. Returns {L1..L3: {node_idx: {rel: {src,attn}}}}."""
    state = _load()
    model = state["model"]
    data  = state["data"]

    x_raw = torch.cat([data["patent"].struct, data["patent"].x], dim=1)
    x = {
        "patent":  model.pat_proj(x_raw),
        "company": F.leaky_relu(model.co_proj(data["company"].x)),
        "ipc":     model.ipc_emb.weight.clone(),
    }
    acts = [F.leaky_relu, F.leaky_relu, torch.tanh]
    all_attn: dict = {}

    for li, (hconv, bn, act) in enumerate(zip(model.convs[:3], model.bns, acts)):
        x_agg: dict = defaultdict(list)
        layer_attn: dict = {}

        for et in EDGE_TYPES:
            s_t, rel, d_t = et
            if et not in data.edge_index_dict or et not in hconv.convs:
                continue
            et_conv   = hconv.convs[et]
            edge_idx  = data.edge_index_dict[et]
            with torch.no_grad():
                out, (ei_ret, alpha) = et_conv(
                    (x[s_t], x[d_t]), edge_idx,
                    return_attention_weights=True)
            for ti in [src_idx, dst_idx]:
                if d_t == "patent":
                    mask = (ei_ret[1] == ti)
                    if mask.any():
                        layer_attn.setdefault(ti, {})[rel] = {
                            "src":  ei_ret[0][mask].cpu().tolist(),
                            "attn": alpha[mask].mean(dim=1).cpu().tolist(),
                        }
            x_agg[d_t].append(out)

        all_attn[f"L{li+1}"] = layer_attn
        x_new = {k: torch.stack(vs).mean(0) for k, vs in x_agg.items()}
        x = {
            k: model.drop(act(bn[k](v.view(-1, model.hid)))) if k in bn
            else model.drop(act(v))
            for k, v in x_new.items()
        }

    return all_attn


# ── Galaxy HTML builder ────────────────────────────────────────────────────────
# 원본 app.py _GALAXY_OPTS / _INJECT_OPTS 그대로
_OPTS_GALAXY = json.dumps({
    "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
            "gravitationalConstant": -55, "centralGravity": 0.003,
            "springLength": 120, "springConstant": 0.04,
            "damping": 0.42, "avoidOverlap": 0.6,
        },
        "stabilization": {"iterations": 280, "updateInterval": 80, "fit": True},
        "maxVelocity": 60, "minVelocity": 0.8,
    },
    "interaction": {"hover": True, "tooltipDelay": 80,
                    "hideEdgesOnDrag": True, "navigationButtons": True},
    "nodes": {"shape": "dot"},
    "edges": {"smooth": {"type": "continuous"}},
})

_OPTS_INJECT = json.dumps({
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

# Original thresholds (same as app.py)
_THR_HIGH = 0.72
_THR_MED  = 0.55


def _build_galaxy_html(
    state:       dict,
    full_graph:  dict,
    query_idx:   int | None = None,
    pred_edges:  list[tuple[int, float]] | None = None,
    gt_indices:  list[int] | None = None,
    hop2_sets:   dict[str, list[int]] | None = None,
    xai_attn:    dict | None = None,
) -> str:
    """Build the single interactive galaxy HTML. Node IDs = p_{idx} (matches app.py)."""
    pl      = state["pat_list"]
    tv_plt  = state["tv_plt_set"]
    tv_def  = state["tv_def_set"]
    lit     = tv_plt | tv_def
    p2i     = state["pat_to_idx"]
    i2i     = state["ipc_to_idx"]
    c2i     = state["co_to_idx"]
    gal     = state["galaxy_pat_set"]

    _gt  = set(gt_indices or [])
    _pe  = dict(pred_edges or [])
    _h2  = hop2_sets or {}

    net = Network(
        height="820px", width="100%",
        bgcolor="#050810", font_color="#FFFFFF",
        directed=True, notebook=False, cdn_resources="in_line",
    )

    # ── Patent nodes (원본 app.py 색상/크기/그림자 그대로) ──────────────────
    all_pat_idx = set(gal)
    if query_idx is not None:
        all_pat_idx.add(query_idx)

    for i in all_pat_idx:
        pid_str  = str(pl[i])
        node_id  = f"p_{i}"
        is_p     = i in tv_plt
        is_d     = i in tv_def
        is_gt    = i in _gt
        is_pred  = i in _pe
        is_query = (i == query_idx)

        if   is_p and is_d: color, sz = "#FFA500", 26
        elif is_p:           color, sz = "#FFD700", 22
        elif is_d:           color, sz = "#FF6B35", 22
        else:                color, sz = "#007B9E", 12

        if is_pred: sz = max(sz, 30)

        role = ("P&D"        if (is_p and is_d)
                else "Plaintiff" if is_p
                else "Defendant" if is_d
                else "•")

        border_col = "#00FF7F" if is_gt else ("#FF0066" if is_pred else color)
        bw         = 4 if is_gt else (3 if is_pred else 1)
        shadow_col = "#00FF7FAA" if is_gt else ("#FF006688" if is_pred else color + "44")
        shadow_sz  = 22 if is_gt else (18 if is_pred else (11 if role != "•" else 4))

        if is_query:
            color, sz  = "#FFFFFF", 56
            border_col = "#00FFFF"
            bw         = 4
            shadow_col = "#FFFFFF99"
            shadow_sz  = 44
            role       = "🛸 QUERY"

        font_col = "#050810" if is_query else ("#555" if role == "•" and not is_pred else "#FFF")
        font_sz  = 14 if is_query else (9 if role == "•" else 11)

        tooltip = (
            f"<b>[PATENT]</b><br>{pid_str}<br>"
            f"<span style='color:{color}'>◉ {role}</span>"
            + (f"<br><b style='color:#00FF7F'>✓ Ground Truth</b>" if is_gt else "")
            + (f"<br><b style='color:#FF0066'>★ 소송 예측 후보</b>" if is_pred else "")
        )

        net.add_node(
            node_id, label=pid_str, title=tooltip,
            color={"background": color, "border": border_col,
                   "highlight": {"background": "#FFF", "border": "#FFF"},
                   "hover":     {"background": "#FFF", "border": "#FFF"}},
            size=sz,
            font={"color": font_col, "size": font_sz, "bold": is_query},
            borderWidth=bw,
            shadow={"enabled": True, "color": shadow_col,
                    "size": shadow_sz, "x": 0, "y": 0},
        )

    # ── Train/Val 소송 엣지 (dark red, 원본 그대로) ──────────────────────────
    sued_arr = np.array(state["ckpt"]["sued_edges"])
    sued_c   = state["ckpt"]["sued_cases"]
    test_cs  = set(state["ckpt"]["test_cases"])
    seen_sued: set = set()
    for i, case_id in enumerate(sued_c):
        if case_id in test_cs:
            continue
        s, d = int(sued_arr[0, i]), int(sued_arr[1, i])
        if s not in gal or d not in gal: continue
        if (s, d) in seen_sued: continue
        seen_sued.add((s, d))
        net.add_edge(
            f"p_{s}", f"p_{d}",
            color={"color": "#8B0000", "hover": "#FF0000"},
            width=2.5,
            arrows={"to": {"enabled": True, "scaleFactor": 0.6}},
            smooth={"type": "curvedCW", "roundness": 0.2},
            title="⚖️ 소송 관계",
        )

    # ── 인용 엣지 (dark blue, 원본 그대로) ───────────────────────────────────
    seen_cit: set = set()
    for _, row in state["df_cit"].iterrows():
        si = p2i.get(str(row.get("citing_patent", row.get("patent_id", ""))))
        di = p2i.get(str(row.get("cited_patent",
                     row.get("citation_patent_id", row.get("cited_id", "")))))
        if si is None or di is None: continue
        if si not in gal or di not in gal: continue
        if (si, di) in seen_cit: continue
        seen_cit.add((si, di))
        net.add_edge(
            f"p_{si}", f"p_{di}",
            color={"color": "#002A44", "hover": "#00BFFF"},
            width=0.9,
            arrows={"to": {"enabled": True, "scaleFactor": 0.35}},
            smooth={"type": "continuous"},
        )

    # ── IPC 노드 + 엣지 (소송특허 + query) ───────────────────────────────────
    tpid_str = str(pl[query_idx]) if query_idx is not None else None
    ipc_g: dict = {}; ipc_e: list = []
    for _, row in state["df_ipc"].iterrows():
        pid_r = str(row.get("patent_id", ""))
        pi    = p2i.get(pid_r)
        ci    = i2i.get(str(row.get("ipc_code", "")))
        if ci is None: continue
        ic = str(row.get("ipc_code", ""))
        if pi is not None and pi in lit and pi in gal:
            ipc_g[ci] = ic; ipc_e.append((f"p_{pi}", f"i_{ci}"))
        if tpid_str and pid_r == tpid_str:
            ipc_g[ci] = ic; ipc_e.append((f"p_{query_idx}", f"i_{ci}"))
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

    # ── Company 노드 + 엣지 (소송특허 + query) ────────────────────────────────
    co_g: dict = {}; co_e: list = []
    for _, row in state["df_asgn"].iterrows():
        pid_r = str(row.get("patent_id", ""))
        pi    = p2i.get(pid_r)
        co_s  = str(row.get("org_norm", row.get("assignee", row.get("company", ""))))
        ci2   = c2i.get(co_s)
        if ci2 is None: continue
        if pi is not None and pi in lit and pi in gal:
            co_g[ci2] = co_s; co_e.append((f"co_{ci2}", f"p_{pi}"))
        if tpid_str and pid_r == tpid_str:
            co_g[ci2] = co_s; co_e.append((f"co_{ci2}", f"p_{query_idx}"))
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

    # ── GT 엣지 (bright green) ────────────────────────────────────────────────
    if query_idx is not None:
        for gt_dst in _gt:
            net.add_edge(
                f"p_{query_idx}", f"p_{gt_dst}",
                color={"color": "#00FF7F", "hover": "#FFFFFF", "highlight": "#FFFFFF"},
                width=4.5,
                arrows={"to": {"enabled": True, "scaleFactor": 0.6}},
                smooth={"type": "curvedCW", "roundness": 0.25},
                title="<b>✓ Ground Truth 소송</b>",
                label="GT",
                font={"color": "#00FF7F", "size": 11, "bold": True, "align": "middle"},
                shadow={"enabled": True, "color": "#00FF7FAA", "size": 22, "x": 0, "y": 0},
            )

    # ── 예측 엣지 (non-GT, 원본 그대로) ──────────────────────────────────────
    if query_idx is not None and pred_edges:
        mx = max(p for _, p in pred_edges) if pred_edges else 1.0
        for rank, (di, prob) in enumerate(pred_edges, 1):
            if di in _gt: continue
            col_c = ("#FF0066" if prob >= _THR_HIGH
                     else "#FF6600" if prob >= _THR_MED
                     else "#FFAA00")
            w = 3.0 + (prob / (mx + 1e-8)) * 12
            net.add_edge(
                f"p_{query_idx}", f"p_{di}",
                label=f"#{rank} {prob:.3f}",
                font={"color": col_c, "size": 12, "bold": True, "align": "middle"},
                title=f"<b>⚠️ 소송 예측</b><br>Cosine: {prob:.4f}  |  Rank: {rank}",
                width=w,
                color={"color": col_c, "highlight": "#FFF", "hover": "#FFF"},
                arrows={"to": {"enabled": False}},
                smooth={"type": "curvedCW", "roundness": 0.25},
                shadow={"enabled": True, "color": col_c + "CC", "size": 22, "x": 0, "y": 0},
            )

    net.set_options(_OPTS_INJECT if query_idx is not None else _OPTS_GALAXY)
    html = net.generate_html()

    # ── JavaScript injection ────────────────────────────────────────────────
    # GT pairs for click detection (node ID format: p_{idx})
    gt_pairs_js: dict = {}
    for gt_dst in _gt:
        gt_pairs_js[f"p_{query_idx}_p_{gt_dst}"] = True
        gt_pairs_js[f"p_{gt_dst}_p_{query_idx}"] = True

    # 2-hop node sets (integer indices → p_{idx} strings)
    hop2_js: dict[str, list[str]] = {}
    for pair_key, idxs in _h2.items():
        parts = pair_key.split("_")
        src_i, dst_i = parts[0], parts[1]
        hop2_js[f"p_{src_i}_p_{dst_i}"] = [f"p_{i}" for i in idxs]

    # XAI attention data (xai_attn = {"{src}_{dst}": {L1:{node:{rel:{src,attn}}}}})
    xai_js: dict = {}
    if xai_attn and query_idx is not None:
        for gt_dst in _gt:
            pair_key_input = f"{query_idx}_{gt_dst}"
            pair_key_js    = f"p_{query_idx}_p_{gt_dst}"
            attn_for_pair  = xai_attn.get(pair_key_input)
            if not attn_for_pair:
                continue
            layer_data: dict = {}
            for layer_name, layer_dict in attn_for_pair.items():   # L1, L2, L3
                node_data: dict = {}
                for node_idx, rel_dict in layer_dict.items():       # int → {rel: {src,attn}}
                    node_data[str(node_idx)] = {
                        rel: {"src": v["src"], "attn": v["attn"]}
                        for rel, v in rel_dict.items()
                    }
                layer_data[layer_name] = node_data
            xai_js[pair_key_js] = layer_data

    js_payload = f"""
<script>
// ── Injected data ──────────────────────────────────────────────
var GT_PAIRS    = {json.dumps(gt_pairs_js)};
var HOP2_DATA   = {json.dumps(hop2_js)};
var XAI_DATA    = {json.dumps(xai_js)};
var QUERY_NODE  = {json.dumps(f"pat_{query_idx}" if query_idx is not None else None)};

// ── Runtime state ──────────────────────────────────────────────
var origNodeColor  = {{}};
var origNodeSize   = {{}};
var origNodeFont   = {{}};
var origEdgeColor  = {{}};
var origEdgeWidth  = {{}};
var edgeByPair     = {{}};   // "pat_A_pat_B" -> edge vis-id
var focusedPair    = null;
var xaiRunning     = false;
var xaiTimeouts    = [];

// ── Init after network ready ────────────────────────────────────
function initGalaxy() {{
    // Save original states
    network.body.data.nodes.getIds().forEach(function(nid) {{
        var n = network.body.data.nodes.get(nid);
        origNodeColor[nid] = n.color   ? JSON.parse(JSON.stringify(n.color)) : {{}};
        origNodeSize[nid]  = n.size    || 10;
        origNodeFont[nid]  = n.font    ? JSON.parse(JSON.stringify(n.font))  : {{}};
    }});
    network.body.data.edges.getIds().forEach(function(eid) {{
        var e = network.body.data.edges.get(eid);
        origEdgeColor[eid] = e.color ? JSON.parse(JSON.stringify(e.color)) : {{}};
        origEdgeWidth[eid] = e.width || 1.0;
        // Build pair→edge mapping
        var k1 = e.from + "_" + e.to;
        var k2 = e.to   + "_" + e.from;
        edgeByPair[k1] = eid;
        edgeByPair[k2] = eid;
    }});

    // Click handler
    network.on("click", function(params) {{
        if (params.edges.length > 0) {{
            var eid       = params.edges[0];
            var connected = network.getConnectedNodes(eid);
            if (connected.length === 2) {{
                var key1 = connected[0] + "_" + connected[1];
                var key2 = connected[1] + "_" + connected[0];
                var pairKey = GT_PAIRS[key1] ? key1 : (GT_PAIRS[key2] ? key2 : null);
                if (pairKey) {{
                    focusSubgraph(pairKey);
                    return;
                }}
            }}
        }}
        // Click on empty → reset
        if (params.nodes.length === 0 && params.edges.length === 0) {{
            resetView();
        }}
    }});

    // Show status bar
    setStatus("Galaxy 로드 완료 · GT 초록 엣지 클릭 → 2-hop 집중", "#00FFFF");
}}

// ── Focus 2-hop subgraph ────────────────────────────────────────
function focusSubgraph(pairKey) {{
    cancelXAI();
    // Normalize pair key to canonical form
    var parts = pairKey.split("_pat_");
    var srcId = parts[0];
    var dstId = "pat_" + parts[1];

    // Try both orientations
    var hop2Ids = HOP2_DATA[pairKey] || HOP2_DATA[dstId + "_" + srcId] || [];
    var hop2Set = new Set(hop2Ids);

    if (hop2Set.size === 0) {{
        setStatus("⚠ 2-hop 데이터 없음", "#FF8800");
        return;
    }}

    focusedPair = pairKey;

    var nodeUpd = [];
    network.body.data.nodes.getIds().forEach(function(nid) {{
        if (hop2Set.has(nid)) {{
            nodeUpd.push({{
                id: nid,
                color: origNodeColor[nid],
                size:  origNodeSize[nid],
                font:  origNodeFont[nid],
            }});
        }} else {{
            nodeUpd.push({{
                id: nid,
                color: {{
                    background: "#0c0c0c", border: "#111",
                    highlight: {{background:"#0c0c0c", border:"#111"}},
                    hover:     {{background:"#0c0c0c", border:"#111"}},
                }},
                size: Math.max(origNodeSize[nid] * 0.25, 2),
                font: {{color:"#111"}},
            }});
        }}
    }});
    network.body.data.nodes.update(nodeUpd);

    var edgeUpd = [];
    network.body.data.edges.getIds().forEach(function(eid) {{
        var e = network.body.data.edges.get(eid);
        var inHop2 = hop2Set.has(e.from) && hop2Set.has(e.to);
        if (inHop2) {{
            edgeUpd.push({{id:eid, color:origEdgeColor[eid], width:origEdgeWidth[eid]}});
        }} else {{
            edgeUpd.push({{
                id: eid,
                color: {{color:"#0c0c0c", hover:"#0c0c0c", highlight:"#0c0c0c"}},
                width: 0.1,
            }});
        }}
    }});
    network.body.data.edges.update(edgeUpd);

    setStatus("🔍 2-hop 집중 · 빈 공간 클릭 = 초기화 · 아래 XAI 버튼으로 분석", "#00FF7F");
    document.getElementById("xai_btn_wrap").style.display = "block";
}}

// ── Reset to full galaxy ────────────────────────────────────────
function resetView() {{
    cancelXAI();
    focusedPair = null;

    var nodeUpd = network.body.data.nodes.getIds().map(function(nid) {{
        return {{id:nid, color:origNodeColor[nid], size:origNodeSize[nid], font:origNodeFont[nid]}};
    }});
    network.body.data.nodes.update(nodeUpd);

    var edgeUpd = network.body.data.edges.getIds().map(function(eid) {{
        return {{id:eid, color:origEdgeColor[eid], width:origEdgeWidth[eid]}};
    }});
    network.body.data.edges.update(edgeUpd);

    document.getElementById("xai_btn_wrap").style.display = "none";
    setStatus("Galaxy 초기화 · GT 초록 엣지 클릭 → 2-hop 집중", "#00FFFF");
    document.getElementById("layer_label").textContent = "";
}}

// ── XAI Layer-by-layer animation ───────────────────────────────
function playXAI() {{
    if (!focusedPair) {{ setStatus("⚠ 먼저 GT 엣지를 클릭하세요", "#FF8800"); return; }}
    var xai = XAI_DATA[focusedPair];
    if (!xai || Object.keys(xai).length === 0) {{
        setStatus("⚠ XAI 데이터 없음 — Streamlit에서 Run XAI 실행 후 다시 시도", "#FF4444");
        return;
    }}
    cancelXAI();
    xaiRunning = true;

    var hop2Ids = HOP2_DATA[focusedPair] || [];
    var hop2Set = new Set(hop2Ids);
    var layers  = ["L1","L2","L3"];
    var delay   = 1400;

    // Reset to 2-hop fade state first
    focusSubgraph(focusedPair);

    layers.forEach(function(layer, li) {{
        var t = setTimeout(function() {{
            if (!xaiRunning) return;
            document.getElementById("layer_label").textContent =
                "▶  Layer " + (li+1) + " / " + layers.length + "  ─  " + layer + " 메시지 전파 중";

            var layerData = xai[layer] || {{}};
            var edgeUpd = [];
            var nodeUpd = [];
            var activeNodes = new Set();

            Object.keys(layerData).forEach(function(nodeIdxStr) {{
                var targetId = "p_" + nodeIdxStr;
                if (!hop2Set.has(targetId)) return;
                activeNodes.add(targetId);

                var relData = layerData[nodeIdxStr];
                Object.keys(relData).forEach(function(rel) {{
                    var srcs  = relData[rel].src  || [];
                    var attns = relData[rel].attn || [];
                    srcs.forEach(function(srcIdx, i) {{
                        var w      = attns[i] || 0;
                        if (w < 0.05) return;
                        var srcId  = "p_" + srcIdx;
                        if (!hop2Set.has(srcId)) return;
                        activeNodes.add(srcId);

                        // Find edge (node IDs use p_ prefix)
                        var eid = edgeByPair["p_" + srcIdx + "_p_" + nodeIdxStr] ||
                                  edgeByPair["p_" + nodeIdxStr + "_p_" + srcIdx];
                        if (eid !== undefined) {{
                            var col = w > 0.65 ? "#FF3333"
                                    : w > 0.40 ? "#FF8800"
                                    : w > 0.20 ? "#FFD700"
                                    : "#AAAAFF";
                            edgeUpd.push({{
                                id:    eid,
                                color: {{color:col, hover:"#FFF", highlight:"#FFF"}},
                                width: 2 + w * 18,
                            }});
                        }}
                    }});
                }});
            }});

            // Restore active nodes from black (node IDs already in p_ format)
            activeNodes.forEach(function(nid) {{
                nodeUpd.push({{
                    id:    nid,
                    color: origNodeColor[nid],
                    size:  origNodeSize[nid] * (1.0 + 0.4 * li),
                    font:  origNodeFont[nid],
                }});
            }});

            if (edgeUpd.length) network.body.data.edges.update(edgeUpd);
            if (nodeUpd.length) network.body.data.nodes.update(nodeUpd);

        }}, delay * (li + 1));
        xaiTimeouts.push(t);
    }});

    // Final state
    var tFinal = setTimeout(function() {{
        if (!xaiRunning) return;
        document.getElementById("layer_label").textContent =
            "✓ 완료 · 밝은 엣지 = High Attention · 클수록 중요 노드";
        xaiRunning = false;
    }}, delay * (layers.length + 1));
    xaiTimeouts.push(tFinal);
}}

function cancelXAI() {{
    xaiTimeouts.forEach(clearTimeout);
    xaiTimeouts = [];
    xaiRunning  = false;
    document.getElementById("layer_label").textContent = "";
}}

// ── UI helpers ──────────────────────────────────────────────────
function setStatus(msg, col) {{
    var el = document.getElementById("status_bar");
    if (el) {{ el.textContent = msg; el.style.color = col; }}
}}

// ── Bootstrap: ForceAtlas2 완전 안정화 후 초기화 ───────────────
(function waitForNetwork() {{
    if (typeof network !== "undefined" && network.body &&
        network.body.data.nodes.length > 0) {{
        // stabilization이 끝나면 initGalaxy, 이미 끝났으면 바로 실행
        if (network.physics && network.physics.stabilized) {{
            initGalaxy();
        }} else {{
            network.once("stabilized", function() {{
                // 물리 끄기 (위치 고정)
                network.setOptions({{physics: {{enabled: false}}}});
                initGalaxy();
            }});
        }}
    }} else {{
        setTimeout(waitForNetwork, 100);
    }}
}})();
</script>

<!-- Overlay UI -->
<div id="galaxy_overlay" style="
    position:absolute; bottom:12px; left:50%;
    transform:translateX(-50%); z-index:200;
    display:flex; flex-direction:column; align-items:center; gap:6px;
    pointer-events:none;">

  <div id="status_bar" style="
      color:#00FFFF; font-family:'JetBrains Mono',monospace;
      font-size:11px; letter-spacing:.06em;
      background:rgba(5,8,16,0.82); border:1px solid #00FFFF33;
      border-radius:6px; padding:5px 14px; pointer-events:none;">
    로딩 중...
  </div>

  <div id="layer_label" style="
      color:#FF8C00; font-family:'JetBrains Mono',monospace;
      font-size:11px; letter-spacing:.04em;
      background:rgba(5,8,16,0.82); border:1px solid #FF8C0033;
      border-radius:6px; padding:4px 12px; min-width:260px;
      text-align:center; pointer-events:none; min-height:24px;">
  </div>

  <div id="xai_btn_wrap" style="display:none; pointer-events:auto;">
    <button onclick="playXAI()" style="
        background:linear-gradient(135deg,#FF006618,#FF440018);
        border:1px solid #FF006699; color:#FF4444;
        border-radius:8px; font-family:'JetBrains Mono',monospace;
        font-weight:700; font-size:12px; letter-spacing:.06em;
        padding:7px 22px; cursor:pointer;
        box-shadow:0 0 10px #FF006633; transition:all .2s;">
      ▶ Play XAI Animation
    </button>
    &nbsp;
    <button onclick="resetView()" style="
        background:rgba(0,255,255,0.05);
        border:1px solid #00FFFF44; color:#00FFFF88;
        border-radius:8px; font-family:'JetBrains Mono',monospace;
        font-size:11px; padding:7px 14px; cursor:pointer; transition:all .2s;">
      ✕ Reset
    </button>
  </div>
</div>
"""

    # Inject overlay into HTML before </body>
    return html.replace("</body>", js_payload + "\n</body>")


# ══════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════
state      = _load()
full_graph = _load_full_graph()
adj        = _build_hetero_adj(0)

pat_list   = state["pat_list"]

# Test plaintiffs from full_graph.json
test_plaintiffs = full_graph.get("test_plaintiffs", [])

# ── Session state ──────────────────────────────────────────────────
if "query_src"   not in st.session_state: st.session_state.query_src   = None
if "xai_attn"    not in st.session_state: st.session_state.xai_attn    = None
if "xai_pair"    not in st.session_state: st.session_state.xai_pair    = None

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='color:#00FFFF;text-shadow:0 0 18px #00FFFF99;"
        "font-size:1.3rem;margin:0 0 2px'>⚖️ Patent Litigation GNN</h2>"
        "<p style='color:#444;font-size:.68rem;margin:0 0 10px'>"
        "SimpleCatHeteroGAT · 4-layer · 8 edge types</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Query patent selector
    st.markdown(
        "<div style='color:#888;font-size:.68rem;letter-spacing:.1em;"
        "text-transform:uppercase;margin-bottom:6px'>🛸 Query Patent (대기실)</div>",
        unsafe_allow_html=True,
    )

    options_labels = ["(전체 Galaxy만 표시)"] + [
        f"{p['label']}  {('· ' + p['owner'][:14]) if p['owner'] else ''}"
        for p in test_plaintiffs
    ]

    sel_in_list = st.selectbox(
        "Query Patent", range(len(options_labels)),
        format_func=lambda i: options_labels[i],
        label_visibility="collapsed",
        key="query_sel",
    )

    if sel_in_list > 0:
        qinfo     = test_plaintiffs[sel_in_list - 1]
        query_idx = qinfo["src_idx"]
        gt_indices = [g["dst_idx"] for g in qinfo["gts"]]
        # Reset XAI if query changed
        if st.session_state.query_src != query_idx:
            st.session_state.query_src = query_idx
            st.session_state.xai_attn  = None
            st.session_state.xai_pair  = None
    else:
        query_idx  = None
        gt_indices = []

    if query_idx is not None:
        # GT 결과: TP/FN + 모델 확률
        h_emb = state["h_emb"]
        pred_set_for_sidebar = {di for di, _ in (pred_edges or [])}
        gt_rows_html = ""
        for g in gt_indices:
            prob_g = float(1 / (1 + np.exp(-float(np.dot(h_emb[query_idx], h_emb[g])))))
            is_tp  = g in pred_set_for_sidebar
            tag_col = "#00C853" if is_tp else "#FF6B35"
            tag_txt = "✓ TP" if is_tp else "✗ FN"
            gt_rows_html += (
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:center;padding:4px 0;border-bottom:1px solid #111'>"
                f"<span style='color:#FF8C00;font-size:.72rem;font-weight:700'>"
                f"→ {pat_list[g]}</span>"
                f"<span style='color:#888;font-size:.68rem'>P={prob_g:.3f}</span>"
                f"<span style='color:{tag_col};font-size:.68rem;font-weight:700;"
                f"border:1px solid {tag_col}55;border-radius:4px;padding:1px 6px'>"
                f"{tag_txt}</span></div>"
            )
        st.markdown(
            f"<div style='background:rgba(0,255,127,0.04);border:1px solid #00FF7F33;"
            f"border-radius:8px;padding:8px 12px;margin:8px 0'>"
            f"<div style='color:#00FFFF;font-size:.75rem;font-weight:700;margin-bottom:6px'>"
            f"🛸 {pat_list[query_idx]}</div>"
            f"{gt_rows_html}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # Top-K slider
        top_k = st.slider("Predicted edges (Top-K)", 5, 30, 20)

        # XAI button
        st.markdown(
            "<div style='color:#888;font-size:.68rem;letter-spacing:.1em;"
            "text-transform:uppercase;margin-bottom:6px'>🧠 XAI Analysis</div>",
            unsafe_allow_html=True,
        )

        xai_target = st.selectbox(
            "GT pair to analyze",
            gt_indices,
            format_func=lambda i: f"{pat_list[query_idx]} → {pat_list[i]}",
        )

        xai_computed = (
            st.session_state.xai_attn is not None and
            st.session_state.xai_pair == (query_idx, xai_target)
        )

        if not xai_computed:
            if st.button("🧠 Compute XAI (Attention)"):
                with st.spinner(f"Attention 계산 중... ({pat_list[query_idx]} → {pat_list[xai_target]})"):
                    st.session_state.xai_attn = _compute_attention(query_idx, xai_target)
                    st.session_state.xai_pair = (query_idx, xai_target)
                st.rerun()
        else:
            st.markdown(
                f"<div style='color:#00FF7F;font-size:.7rem;padding:4px 0'>"
                f"✓ XAI ready: {pat_list[query_idx]} → {pat_list[xai_target]}<br>"
                f"<span style='color:#555'>Galaxy에서 GT 엣지 클릭 후 ▶ Play</span></div>",
                unsafe_allow_html=True,
            )
            if st.button("🔄 재계산"):
                st.session_state.xai_attn = None
                st.session_state.xai_pair = None
                st.rerun()
    else:
        top_k = 20
        xai_target = None

    st.divider()
    st.markdown(
        "<div style='color:#333;font-size:.6rem;line-height:1.7'>"
        "● GT 초록 엣지 클릭 → 2-hop 집중<br>"
        "● 빈 공간 클릭 → 전체 초기화<br>"
        "● XAI 계산 후 ▶ Play → 레이어 애니메이션"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Compute predictions ────────────────────────────────────────────
pred_edges: list[tuple[int, float]] = []
if query_idx is not None:
    from backend.loader import predict_top_k
    pred_edges = predict_top_k(state["h_emb"], query_idx, k=top_k)

# ── Compute 2-hop sets ─────────────────────────────────────────────
hop2_sets: dict[str, list[int]] = {}
if query_idx is not None:
    for gt_dst in gt_indices:
        pair_key = f"{query_idx}_{gt_dst}"
        with st.spinner(f"2-hop 계산 중 ({pat_list[query_idx]} → {pat_list[gt_dst]})..."):
            hop2_sets[pair_key] = _hop2_patents(adj, query_idx, gt_dst)

# ── XAI data for injection ─────────────────────────────────────────
xai_for_html: dict | None = None
if (st.session_state.xai_attn is not None and
        st.session_state.xai_pair is not None and
        st.session_state.xai_pair[0] == query_idx):
    xs, xd = st.session_state.xai_pair
    pair_key_for_html = f"{xs}_{xd}"
    # wrap in {pair_key: attn_data}
    xai_for_html = {pair_key_for_html: st.session_state.xai_attn}

# ── Main panel header ──────────────────────────────────────────────
col_h, col_info = st.columns([3, 1])
with col_h:
    st.markdown(
        "<h1 style='color:#00FFFF;text-shadow:0 0 24px #00FFFF66;"
        "font-size:1.6rem;margin:0 0 2px;font-family:JetBrains Mono,monospace'>"
        "⚖️ Patent Litigation Galaxy</h1>",
        unsafe_allow_html=True,
    )
with col_info:
    c1, c2, c3 = st.columns(3)
    c1.metric("Patents", "1,331")
    c2.metric("GT Pairs", "251")
    c3.metric("Selected", pat_list[query_idx] if query_idx else "—")

# ── Build & render galaxy ──────────────────────────────────────────
with st.spinner("Galaxy 빌드 중..."):
    galaxy_html = _build_galaxy_html(
        state       = state,
        full_graph  = full_graph,
        query_idx   = query_idx,
        pred_edges  = pred_edges,
        gt_indices  = gt_indices,
        hop2_sets   = hop2_sets,
        xai_attn    = xai_for_html,
    )

components.html(galaxy_html, height=900, scrolling=False)

# ── Legend ─────────────────────────────────────────────────────────
st.markdown(
    "<div style='color:#333;font-size:.65rem;text-align:center;margin-top:4px;"
    "font-family:JetBrains Mono,monospace'>"
    "<span style='color:#FFD700'>● Plaintiff</span> &nbsp;"
    "<span style='color:#FF6B35'>● Defendant</span> &nbsp;"
    "<span style='color:#FFA500'>● P&D</span> &nbsp;"
    "<span style='color:#007B9E'>● Other</span> &nbsp;|&nbsp;"
    "<span style='color:#FFFFFF'>● Query</span> &nbsp;"
    "<span style='color:#00FF7F'>● GT</span> &nbsp;"
    "<span style='color:#FF0066'>★ Pred-H</span> &nbsp;"
    "<span style='color:#FF6600'>★ Pred-M</span> &nbsp;"
    "<span style='color:#FFAA00'>★ Pred-L</span>"
    "</div>",
    unsafe_allow_html=True,
)
