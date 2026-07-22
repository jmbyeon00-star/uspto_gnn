"""
xai_viz_app.py  v3
Patent Litigation XAI — Streamlit Interactive Demo

Tabs:
  📊 Subgraph         — GNNExplainer mask, fixed layout
  🧠 Attention Flow   — per-layer native GAT attention + Evidence Flow animation
  ⊕  Overlay          — Attention / GNNExplainer / Counterfactual 3-way comparison
  🔗 Evidence Path    — path visualization
  🔄 Counterfactual   — edge type + top-K analysis
  📋 Agreement        — scatter + Pearson/Spearman
  ℹ️ Storyline         — 4-layer explanation narrative + gauge

Run:
    streamlit run xai_viz_app.py
"""

import json, os, tempfile, time
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from pyvis.network import Network

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patent Litigation XAI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""<style>
  .block-container { padding-top: 0.7rem; }
  div[data-testid="metric-container"] { border: 1px solid #333; border-radius:8px; padding:6px; }
</style>""", unsafe_allow_html=True)

# ─── Load JSON ────────────────────────────────────────────────────────────────
JSON_PATH = os.path.join(os.path.dirname(__file__), "xai_results.json")

@st.cache_data
def load_results(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

if not os.path.exists(JSON_PATH):
    st.error("먼저 `python xai_compute.py`를 실행하세요.")
    st.stop()

res       = load_results(JSON_PATH)
meta      = res["meta"]
nodes     = res["nodes"]
edges     = res["edges"]
cf        = res["counterfactual"]
ep        = res.get("evidence_path", {"nodes": [], "edges": [], "score": 0})
agreement = res.get("agreement", [])
storyline = res.get("storyline", [])
p_base    = meta["p_base"]

node_by_id = {n["id"]: n for n in nodes}

LAYER_NAMES = ["layer0", "layer1", "layer2", "layer3"]
LAYER_LABELS = {"layer0": "Layer 1 (×8h)", "layer1": "Layer 2 (×4h)",
                "layer2": "Layer 3 (×4h)", "layer3": "Layer 4 (×2h)"}

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Patent XAI")
    st.markdown("---")

    pc = "#ff4b4b" if p_base > 0.7 else "#ffa500" if p_base > 0.5 else "#44aa44"
    st.markdown(
        f"**Plaintiff**\n\n`{meta['src_label']}`\n\n"
        f"**Defendant**\n\n`{meta['dst_label']}`\n\n"
        f"P(sue) = **<span style='color:{pc};font-size:1.4em'>{p_base:.4f}</span>**",
        unsafe_allow_html=True,
    )
    st.caption("🆕 New" if meta.get("is_new_patent") else "📁 Known")

    st.markdown("---")
    st.markdown("### Controls")
    threshold    = st.slider("Edge mask threshold", 0.0, 1.0, 0.05, 0.01)
    top_k_show   = st.slider("Top-K highlight", 1, 20, meta.get("top_k_cf", 10))
    node_scale   = st.slider("Node size scale", 10, 60, 25)
    show_path    = st.checkbox("Highlight evidence path", value=True)

    st.markdown("---")
    st.markdown("### Visibility")
    show_patent  = st.checkbox("Patent",  value=True)
    show_company = st.checkbox("Company", value=True)
    show_ipc     = st.checkbox("IPC",     value=False)
    show_cites   = st.checkbox("cites/cited_by", value=True)
    show_sued    = st.checkbox("sued/sued_by",   value=True)
    show_co_e    = st.checkbox("owned_by",        value=True)
    show_ipc_e   = st.checkbox("has_ipc",         value=False)

    st.markdown("---")
    lang     = st.radio("Language", ["한국어", "English"], horizontal=True)
    lang_key = "ko" if lang == "한국어" else "en"

    st.caption(f"K={meta['k_hop']} | N={meta['n_nodes']} | E={meta['n_edges']}")

# ─── Helpers ──────────────────────────────────────────────────────────────────
CITE_TYPES = {"cites", "cited_by"}
SUED_TYPES = {"sued", "sued_by"}
NODE_COLOR = {"patent": "#aaaaaa", "company": "#44cc88", "ipc": "#f0a020"}
NODE_SHAPE = {"patent": "dot",      "company": "triangle", "ipc": "hexagon"}
EDGE_COLOR = {
    "cites": "#f0c040", "cited_by": "#f08040",
    "sued":  "#ff4444", "sued_by":  "#ff8888",
    "owned_by": "#44cc88", "has_ipc": "#f0a020",
}

def keep_edge(e):
    if e["type"] not in ("owned_by", "has_ipc") and e["gnn_mask"] < threshold:
        return False
    if e["type"] in CITE_TYPES and not show_cites: return False
    if e["type"] in SUED_TYPES and not show_sued:  return False
    if e["type"] == "owned_by"  and not show_co_e: return False
    if e["type"] == "has_ipc"   and not show_ipc_e: return False
    return True

def keep_node(n):
    if n["type"] == "patent"  and not show_patent:  return False
    if n["type"] == "company" and not show_company: return False
    if n["type"] == "ipc"     and not show_ipc:     return False
    return True

filtered_edges = [e for e in edges if keep_edge(e)]
top_k_ids      = {e["id"] for e in sorted(edges, key=lambda e: -e["gnn_mask"])[:top_k_show]}
path_node_set  = set(ep.get("nodes", []))
path_edge_pairs = {(pe["src"], pe["dst"]) for pe in ep.get("edges", [])}

def _score_to_hex(v: float, method: str) -> str:
    """Map a [0,1] score to a hex color per method."""
    v = max(0.0, min(1.0, v))
    if method == "attn":
        b = int(60 + 195 * v); return f"#{30:02x}{80:02x}{b:02x}"
    if method == "gnn":
        r = int(60 + 195 * v); return f"#{r:02x}{30:02x}{30:02x}"
    if method == "cf":
        g = int(60 + 195 * v); return f"#{30:02x}{g:02x}{30:02x}"
    return "#888888"

def build_pyvis(
    height="600px",
    mode="gnn",          # "gnn" | "attn_L0".."attn_L3" | "overlay" | "flow"
    flow_hop=99,         # for mode="flow": show nodes with min_hop <= flow_hop
    path_only=False,
) -> str:
    net = Network(
        height=height, width="100%",
        bgcolor="#0d1117", font_color="#eeeeee",
        directed=True, notebook=False,
    )
    net.set_options("""{
      "physics": {"enabled": false},
      "edges": {
        "smooth": {"type":"curvedCW","roundness":0.1},
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.45}},
        "scaling": {"min":1,"max":14}
      },
      "nodes": {"borderWidth":2,"shadow":{"enabled":true}},
      "interaction": {"hover":true,"navigationButtons":true,"zoomView":true}
    }""")

    # Determine which layer for attention
    attn_layer = None
    if mode.startswith("attn_L"):
        attn_layer = f"layer{mode[-1]}"

    # Which nodes to show
    if path_only:
        show_nids = path_node_set
    elif mode == "flow":
        show_nids = {n["id"] for n in nodes
                     if n["min_hop"] <= flow_hop and keep_node(n)}
    else:
        used = set()
        for e in filtered_edges:
            ns, nd = node_by_id.get(e["src"]), node_by_id.get(e["dst"])
            if ns and keep_node(ns): used.add(e["src"])
            if nd and keep_node(nd): used.add(e["dst"])
        show_nids = used

    # Add nodes
    for nid in show_nids:
        n = node_by_id.get(nid)
        if n is None: continue
        on_path = show_path and nid in path_node_set

        if n["is_src"]:
            nc, shape, sz = "#ff4b4b", "star",    node_scale * 1.8
        elif n["is_dst"]:
            nc, shape, sz = "#1e90ff", "diamond", node_scale * 1.5
        elif on_path:
            nc = "#ffdd00"; shape = NODE_SHAPE.get(n["type"], "dot")
            sz = max(10, n["importance"] * node_scale * 1.2)
        else:
            nc    = NODE_COLOR.get(n["type"], "#aaaaaa")
            shape = NODE_SHAPE.get(n["type"], "dot")
            sz    = max(6, n["importance"] * node_scale)

        bc    = "#ffdd00" if (on_path and not n["is_src"] and not n["is_dst"]) else nc
        hop_s = n.get("hop_from_src", "?"); hop_d = n.get("hop_from_dst", "?")
        tooltip = (
            f"<b>{n['label']}</b><br>"
            f"Type: {n['type']}<br>"
            f"Importance: {n['importance']:.3f}<br>"
            f"Hop (src/dst): {hop_s}/{hop_d}"
            + ("<br>⚖️ Plaintiff"      if n["is_src"] else
               "<br>🛡️ Defendant"     if n["is_dst"] else
               "<br>🔗 Evidence Path"  if on_path    else "")
        )
        net.add_node(
            nid, x=n["x"], y=n["y"], physics=False,
            label=n["short_label"],
            color={"background": nc, "border": bc},
            shape=shape, size=sz, title=tooltip,
            font={"size": 9, "color": "#eeeeee"},
        )

    PAT_PAT_TYPES = {"cites", "cited_by", "sued", "sued_by"}

    # Which edges to show
    if path_only:
        disp_edges = [e for e in edges if (e["src"], e["dst"]) in path_edge_pairs]
    elif mode == "flow":
        disp_edges = [e for e in edges
                      if e["src"] in show_nids and e["dst"] in show_nids
                      and e["type"] in PAT_PAT_TYPES]
    else:
        disp_edges = filtered_edges

    for e in disp_edges:
        if e["src"] not in show_nids or e["dst"] not in show_nids:
            continue
        is_top  = e["id"] in top_k_ids
        on_pe   = show_path and (e["src"], e["dst"]) in path_edge_pairs

        # Width and color per mode
        if mode == "gnn":
            score = e["gnn_mask"]
            ec    = EDGE_COLOR.get(e["type"], "#888888")
        elif attn_layer:
            score = e.get("attention", {}).get(attn_layer, 0.0)
            ec    = _score_to_hex(score, "attn")
        elif mode == "overlay":
            attn_v = e.get("attention", {}).get("layer2", 0.0)
            gnn_v  = e["gnn_mask"]
            cf_v   = abs(e["delta_p"]) / 0.5 if e.get("delta_p") is not None else 0.0
            cf_v   = min(cf_v, 1.0)
            # RGB blend: R=GNN, G=CF, B=Attn
            r = int(40 + 200 * gnn_v)
            g = int(40 + 200 * cf_v)
            b = int(40 + 200 * attn_v)
            ec    = f"#{r:02x}{g:02x}{b:02x}"
            score = max(attn_v, gnn_v, cf_v)
        elif mode == "flow":
            ln    = f"layer{min(flow_hop, 3)}"
            score = e.get("attention", {}).get(ln, 0.0)
            ec    = _score_to_hex(score, "attn")
        else:
            score = e["gnn_mask"]
            ec    = EDGE_COLOR.get(e["type"], "#888888")

        if on_pe: ec = "#ffdd00"
        width = max(1.0, score * 12) if score > 0 else 1.5

        dp_str   = f"{e['delta_p']:+.4f}" if e.get("delta_p") is not None else "—"
        attn_str = ", ".join(
            f"{LAYER_LABELS.get(ln,'?')}={v:.3f}"
            for ln, v in (e.get("attention") or {}).items()
        ) or "—"
        rank_str = f"#{e['rank']}" if e.get("rank") else "—"
        tooltip  = (
            f"Type: <b>{e['type']}</b><br>"
            f"GNN mask: {e['gnn_mask']:.4f}  Rank: {rank_str}<br>"
            f"Counterfactual ΔP: {dp_str}<br>"
            f"Attention: {attn_str}"
            + ("<br>⭐ Top-K"         if is_top else "")
            + ("<br>🔗 Evidence Path" if on_pe  else "")
        )
        net.add_edge(
            e["src"], e["dst"], width=width,
            color={"color": ec, "opacity": 0.92 if (is_top or on_pe) else 0.42},
            title=tooltip, dashes=not (is_top or on_pe),
        )

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False,
                                     mode="w", encoding="utf-8") as tmp:
        net.save_graph(tmp.name)
        html = open(tmp.name, encoding="utf-8").read()
    os.unlink(tmp.name)
    return html

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Subgraph",
    "🧠 Attention Flow",
    "⊕ Overlay",
    "🔗 Evidence Path",
    "🔄 Counterfactual",
    "📋 Agreement",
    "ℹ️ Storyline",
])
tab_graph, tab_attn, tab_overlay, tab_path, tab_cf, tab_agree, tab_story = tabs

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 · Subgraph (GNNExplainer)
# ══════════════════════════════════════════════════════════════════════════════
with tab_graph:
    col_g, col_legend = st.columns([5, 1])

    with col_legend:
        st.markdown("**Nodes**")
        st.markdown("🔴 Plaintiff ★"); st.markdown("🔵 Defendant ◆")
        st.markdown("⚪ Patent ●");    st.markdown("🟢 Company ▲")
        st.markdown("🟡 IPC ⬡");       st.markdown("🔶 Path")
        st.markdown("---")
        st.markdown("**Edges**")
        for t, c in EDGE_COLOR.items():
            st.markdown(f"<span style='color:{c}'>■</span> {t}", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Width** = GNN mask")
        st.markdown("**Size** = Importance")

    with col_g:
        components.html(build_pyvis("610px", mode="gnn"), height=630, scrolling=False)

    vis_nids = set()
    for e in filtered_edges:
        vis_nids.add(e["src"]); vis_nids.add(e["dst"])
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Nodes", len(vis_nids))
    c2.metric("Edges", len(filtered_edges))
    c3.metric("Top-K", sum(1 for e in filtered_edges if e["id"] in top_k_ids))
    c4.metric("Path length", len(ep.get("nodes", [])))
    c5.metric("Path score", f"{ep.get('score', 0):.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 · Attention Flow
# ══════════════════════════════════════════════════════════════════════════════
with tab_attn:
    st.subheader("🧠 Native GAT Attention — per layer")

    col_ctrl, col_info = st.columns([3, 2])
    with col_ctrl:
        attn_mode = st.radio(
            "Display mode",
            ["Layer 1", "Layer 2", "Layer 3", "Layer 4", "▶ Evidence Flow"],
            horizontal=True,
        )
    with col_info:
        attn_avg = meta.get("attn_layer_avg", {})
        st.markdown("**Top-K avg attention per layer**")
        cols_l = st.columns(4)
        for i, ln in enumerate(LAYER_NAMES):
            cols_l[i].metric(f"L{i+1}", f"{attn_avg.get(ln, 0):.3f}")

    # ── Evidence Flow animation ──────────────────────────────────────────────
    if attn_mode == "▶ Evidence Flow":
        FLOW_STEPS = [
            (0, "Hop 0 · Lawsuit Pair Only"),
            (1, "Hop 1 · Layer 1 Attention"),
            (2, "Hop 2 · Layer 2 Attention"),
            (3, "Hop 3 · Layer 3 Attention"),
            (4, "Full View · Layer 4 Attention"),
        ]

        if "flow_step" not in st.session_state:
            st.session_state.flow_step = 0
        if "flow_playing" not in st.session_state:
            st.session_state.flow_playing = False

        step = st.session_state.flow_step
        c_prev, c_label, c_next, c_play = st.columns([1, 4, 1, 1])

        if c_prev.button("◀", key="f_prev"):
            st.session_state.flow_step = max(0, step - 1)
            st.session_state.flow_playing = False

        c_label.markdown(
            f"**Step {step+1}/{len(FLOW_STEPS)}:**  {FLOW_STEPS[step][1]}"
        )

        if c_next.button("▶", key="f_next"):
            st.session_state.flow_step = min(len(FLOW_STEPS)-1, step + 1)
            st.session_state.flow_playing = False

        play_label = "⏸" if st.session_state.flow_playing else "Auto"
        if c_play.button(play_label, key="f_play"):
            st.session_state.flow_playing = not st.session_state.flow_playing

        hop_lim = FLOW_STEPS[step][0]
        components.html(
            build_pyvis("560px", mode="flow", flow_hop=hop_lim),
            height=580, scrolling=False,
        )

        # Auto-advance
        if st.session_state.flow_playing:
            time.sleep(1.6)
            nxt = step + 1
            if nxt >= len(FLOW_STEPS):
                nxt = 0
                st.session_state.flow_playing = False
            st.session_state.flow_step = nxt
            st.rerun()

        st.progress((step + 1) / len(FLOW_STEPS))
        st.caption(
            "각 step은 GNN이 해당 레이어에서 information을 집계하는 범위를 보여줍니다. "
            "엣지 색상 = attention weight (파란색 밝을수록 높음)."
        )

    # ── Per-layer static view ────────────────────────────────────────────────
    else:
        layer_map = {
            "Layer 1": "attn_L0", "Layer 2": "attn_L1",
            "Layer 3": "attn_L2", "Layer 4": "attn_L3",
        }
        pyvis_mode = layer_map[attn_mode]
        components.html(build_pyvis("560px", mode=pyvis_mode), height=580, scrolling=False)

        # Attention distribution bar chart
        ln_key = f"layer{pyvis_mode[-1]}"
        attn_vals = [
            e["attention"].get(ln_key, 0.0)
            for e in edges
            if e.get("attention") and e["attention"].get(ln_key) is not None
        ]
        if attn_vals:
            fig_attn = go.Figure(go.Histogram(
                x=attn_vals, nbinsx=30,
                marker_color="#4488ff", opacity=0.8,
            ))
            fig_attn.update_layout(
                title=f"{attn_mode} attention distribution (subgraph edges)",
                xaxis_title="Attention weight", yaxis_title="Count",
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                font_color="white", height=220,
            )
            st.plotly_chart(fig_attn, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 · Overlay  (Attention + GNNExplainer + Counterfactual)
# ══════════════════════════════════════════════════════════════════════════════
with tab_overlay:
    st.subheader("⊕ Three-Method Overlay")
    st.markdown("""
| Channel | Method | Meaning |
|---|---|---|
| 🔵 Blue saturation | Native Attention (Layer 2) | 모델이 직접 학습한 중요도 |
| 🔴 Red saturation  | GNNExplainer mask         | 사후 설명 |
| 🟢 Green saturation | Counterfactual \|ΔP\|    | 인과적 영향 |
> 흰색에 가까울수록 세 방법 모두 중요하다고 동의한 엣지입니다.
    """)

    components.html(build_pyvis("580px", mode="overlay"), height=600, scrolling=False)

    # Legend bar
    fig_leg = go.Figure()
    for label, color in [("Attention", "#2244ff"), ("GNNExplainer", "#ff2244"),
                          ("Counterfactual", "#22ff44"), ("All agree", "#ffffff")]:
        fig_leg.add_trace(go.Bar(
            x=[label], y=[1], marker_color=color,
            showlegend=False, text=[label], textposition="inside",
        ))
    fig_leg.update_layout(
        barmode="group", height=80,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        margin=dict(t=5, b=5, l=5, r=5),
    )
    st.plotly_chart(fig_leg, use_container_width=True)

    # Scatter: attention vs GNN mask
    attn_L2   = [e.get("attention", {}).get("layer2", 0) for e in edges if e.get("attention")]
    gnn_masks = [e["gnn_mask"] for e in edges if e.get("attention")]
    cf_dps    = [abs(e.get("delta_p") or 0) for e in edges if e.get("attention")]

    if attn_L2:
        fig_sc = go.Figure(go.Scatter(
            x=attn_L2, y=gnn_masks,
            mode="markers",
            marker=dict(
                size=[6 + d * 30 for d in cf_dps],
                color=[f"rgb({int(40+200*g)},{int(40+200*d)},{int(40+200*a)})"
                       for a, g, d in zip(attn_L2, gnn_masks, cf_dps)],
                opacity=0.8,
                line=dict(color="white", width=0.5),
            ),
            hovertemplate="Attn=%{x:.3f}<br>GNN=%{y:.3f}<extra></extra>",
        ))
        fig_sc.update_layout(
            title="Attention (L2) vs GNN mask  [size = |ΔP|]",
            xaxis_title="Attention (Layer 2)", yaxis_title="GNNExplainer mask",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="white", height=320,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 · Evidence Path
# ══════════════════════════════════════════════════════════════════════════════
with tab_path:
    st.subheader("🔗 Evidence Path")
    path_nodes_list = ep.get("nodes", [])
    path_edges_list = ep.get("edges", [])

    if not path_nodes_list:
        st.warning("경로 없음.")
    else:
        # Flow diagram
        step_labels, step_colors = [], []
        for nid in path_nodes_list:
            n   = node_by_id.get(nid, {})
            lbl = n.get("label", nid)[:22]
            nt  = n.get("type", "?")
            icon = "⚖️" if n.get("is_src") else "🛡️" if n.get("is_dst") else \
                   "🏢" if nt == "company" else "🔖" if nt == "ipc" else "📄"
            step_labels.append(f"{icon} {lbl}")
            step_colors.append(
                "#ff4b4b" if n.get("is_src") else
                "#1e90ff" if n.get("is_dst") else "#ffdd00"
            )

        n_s = len(step_labels)
        fig_flow = go.Figure()
        for i, (lbl, col) in enumerate(zip(step_labels, step_colors)):
            fig_flow.add_trace(go.Scatter(
                x=[i], y=[0], mode="markers+text",
                text=[lbl], textposition="bottom center",
                marker=dict(size=28, color=col, symbol="circle"),
                showlegend=False,
            ))
        for i in range(n_s - 1):
            pe  = path_edges_list[i] if i < len(path_edges_list) else {}
            gm  = pe.get("gnn_mask", 0); atv = pe.get("attn", 0)
            fig_flow.add_annotation(
                x=i+0.5, y=0.2,
                text=f"{pe.get('type','')}<br>mask={gm:.2f} attn={atv:.2f}",
                showarrow=False, font=dict(size=8, color="#aaaaaa"),
            )
            fig_flow.add_annotation(
                ax=i, ay=0, x=i+1, y=0,
                xref="x", yref="y", axref="x", ayref="y",
                arrowhead=2, arrowcolor="#ffdd00", arrowwidth=2,
            )
        fig_flow.update_layout(
            xaxis=dict(range=[-0.5, n_s-0.5], visible=False),
            yaxis=dict(range=[-0.6, 0.5], visible=False),
            paper_bgcolor="#0d1117", height=200,
            margin=dict(t=10, b=70, l=10, r=10),
        )
        st.plotly_chart(fig_flow, use_container_width=True)
        st.caption(f"Path score: **{ep.get('score', 0):.4f}**")

        st.markdown("---")
        if path_node_set:
            components.html(build_pyvis("340px", path_only=True), height=360)

        st.dataframe(
            [{"Step": i+1, "ID": nid,
              "Type": node_by_id.get(nid, {}).get("type", "?"),
              "Label": node_by_id.get(nid, {}).get("label", nid)[:35],
              "Edge →": (path_edges_list[i]["type"] if i < len(path_edges_list) else ""),
              "GNN mask": (f"{path_edges_list[i]['gnn_mask']:.4f}" if i < len(path_edges_list) else ""),
              "Attention": (f"{path_edges_list[i]['attn']:.4f}" if i < len(path_edges_list) else ""),
              } for i, nid in enumerate(path_nodes_list)],
            use_container_width=True, hide_index=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 · Counterfactual
# ══════════════════════════════════════════════════════════════════════════════
with tab_cf:
    st.subheader("어떤 관계를 끊으면 소송 예측이 가장 많이 떨어지나?")
    col_t, col_e = st.columns(2)

    with col_t:
        st.markdown("#### 엣지 타입별")
        td = cf["by_type"]
        et_n = list(td.keys()); deltas = [td[k]["delta_p"] for k in et_n]
        p_afts = [td[k]["p_after"] for k in et_n]
        bc = ["#ff2222" if d<-0.10 else "#ff7744" if d<-0.01 else "#888" for d in deltas]
        fig_t = go.Figure(go.Bar(x=et_n, y=deltas, marker_color=bc,
            text=[f"Δ{d:+.3f}" for d in deltas], textposition="outside"))
        fig_t.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
        fig_t.update_layout(title=f"ΔP (base={p_base:.4f})",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="white", height=360)
        st.plotly_chart(fig_t, use_container_width=True)
        mit = min(td, key=lambda k: td[k]["delta_p"])
        st.info(f"🎯 **{mit}** 제거 시 Δ = `{td[mit]['delta_p']:+.4f}`")

    with col_e:
        st.markdown("#### GNNExplainer Top-K")
        ed = cf["by_edge"]; e_d = [e["delta_p"] for e in ed]
        e_bc = ["#ff2222" if d<-0.05 else "#ff8844" if d<0 else "#44aa44" for d in e_d]
        fig_e = go.Figure(go.Bar(x=list(range(len(ed))), y=e_d, marker_color=e_bc,
            text=[f"{d:+.3f}" for d in e_d], textposition="outside"))
        fig_e.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
        fig_e.update_layout(
            title="ΔP per removed edge",
            xaxis=dict(tickvals=list(range(len(ed))),
                       ticktext=[f"#{e['rank']} {e['edge_type']}" for e in ed],
                       tickangle=45),
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="white", height=360)
        st.plotly_chart(fig_e, use_container_width=True)
        st.dataframe(
            [{"Rank": e["rank"], "Type": e["edge_type"],
              "Src": e["src_label"][:20], "Dst": e["dst_label"][:20],
              "GNN mask": f"{e['gnn_mask']:.3f}",
              "P after": f"{e['p_after']:.4f}", "ΔP": f"{e['delta_p']:+.4f}"}
             for e in ed],
            use_container_width=True, hide_index=True)

    st.markdown("---"); st.markdown("#### Before / After: Top-3")
    top3 = sorted(ed, key=lambda e: e["delta_p"])[:3]
    x_ba = [f"Base\n{p_base:.3f}"] + [f"#{e['rank']} {e['edge_type'][:5]}\n{e['p_after']:.3f}" for e in top3]
    y_ba = [p_base] + [e["p_after"] for e in top3]
    c_ba = ["#44aaff"] + ["#ff2222" if v<p_base*0.7 else "#ff8844" for v in [e["p_after"] for e in top3]]
    fig_ba = go.Figure(go.Bar(x=x_ba, y=y_ba, marker_color=c_ba,
        text=[f"{v:.4f}" for v in y_ba], textposition="outside"))
    fig_ba.add_hline(y=0.5, line_dash="dot", line_color="yellow", opacity=0.5,
                     annotation_text="threshold=0.5", annotation_font_color="yellow")
    fig_ba.update_layout(yaxis=dict(range=[0, 1.15], title="P(sue)"),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="white", height=280)
    st.plotly_chart(fig_ba, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 6 · Agreement  (+Pearson/Spearman)
# ══════════════════════════════════════════════════════════════════════════════
with tab_agree:
    st.subheader("GNNExplainer Importance vs Counterfactual ΔP")

    if not agreement:
        st.warning("agreement 데이터 없음.")
    else:
        imp_v = [a["gnn_importance"] for a in agreement]
        dp_v  = [a["delta_p"]        for a in agreement]

        # Correlation
        try:
            from scipy import stats as sp_stats
            r_p, p_p = sp_stats.pearsonr(imp_v, dp_v)
            r_s, p_s = sp_stats.spearmanr(imp_v, dp_v)
            cp1, cp2 = st.columns(2)
            cp1.metric("Pearson r",   f"{r_p:.3f}",  f"p={p_p:.3f}")
            cp2.metric("Spearman ρ",  f"{r_s:.3f}",  f"p={p_s:.3f}")
            if abs(r_p) > 0.6:
                st.success("두 방법 강한 일치 → 설명의 신뢰도 높음.")
            elif abs(r_p) > 0.3:
                st.info("두 방법 중간 수준 일치.")
            else:
                st.warning("두 방법 약한 일치 — 주의 필요.")
        except ImportError:
            st.caption("scipy 없음 — 상관계수 계산 불가.")

        agree_c = ["#44ff88" if a["agree"] else "#ff4444" for a in agreement]
        labels  = [f"#{a['rank']} {a['edge_type']}" for a in agreement]

        fig_sc = go.Figure(go.Scatter(
            x=imp_v, y=dp_v, mode="markers+text",
            text=labels, textposition="top center", textfont=dict(size=8),
            marker=dict(size=12, color=agree_c, line=dict(color="white", width=1)),
        ))
        fig_sc.add_hline(y=0,   line_dash="dash", line_color="white", opacity=0.3)
        fig_sc.add_vline(x=0.4, line_dash="dash", line_color="white", opacity=0.3,
                         annotation_text="threshold 0.4")
        fig_sc.update_layout(
            title="GNN Importance vs ΔP  [🟢 agree | 🔴 disagree]",
            xaxis_title="GNN Importance", yaxis_title="ΔP(sue)",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="white", height=400,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        st.dataframe(
            [{"Rank": a["rank"], "Type": a["edge_type"],
              "Src": a["src_label"][:20], "Dst": a["dst_label"][:20],
              "GNN Imp.": f"{a['gnn_importance']:.3f}",
              "ΔP": f"{a['delta_p']:+.4f}",
              "Agree": "✅" if a["agree"] else "❌"}
             for a in agreement],
            use_container_width=True, hide_index=True,
        )
        agree_cnt = sum(1 for a in agreement if a["agree"])
        st.info(f"일치: **{agree_cnt}/{len(agreement)}** ({100*agree_cnt//max(len(agreement),1)}%)")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 7 · Storyline
# ══════════════════════════════════════════════════════════════════════════════
with tab_story:
    col_meta, col_gauge = st.columns([2, 1])

    with col_meta:
        st.subheader("Why was this lawsuit predicted?")

        if meta.get("is_new_patent"):
            st.success("🆕 New Patent Scenario")
            c1, c2, c3 = st.columns(3)
            c1.metric("Patent",      meta.get("query_patent", "?")[:18])
            c2.metric("Citations",   meta.get("src_citations", "?"))
            c3.metric("Owner",       (", ".join(meta.get("src_owner_companies", [])))[:20] or "?")
            st.markdown(f"**IPC**: `{', '.join(meta.get('src_ipc_codes', [])[:4]) or '?'}`")

            # New patent insertion diagram
            st.markdown("#### New Patent → Graph Connection")
            ins_labels = [
                f"📄 {meta.get('query_patent', '?')[:20]}",
                f"🏢 {(meta.get('src_owner_companies') or ['?'])[0][:18]}",
                f"🔖 {(meta.get('src_ipc_codes') or ['?'])[0][:10]}",
                "🕸️ Existing Graph",
                f"⚖️ P(sue)={p_base:.3f}",
            ]
            ins_x = list(range(len(ins_labels)))
            fig_ins = go.Figure()
            for i, lbl in enumerate(ins_labels):
                fig_ins.add_trace(go.Scatter(
                    x=[i], y=[0], mode="markers+text",
                    text=[lbl], textposition="bottom center",
                    marker=dict(size=24, color=["#44aaff","#44cc88","#f0a020","#aaaaaa","#ff4b4b"][i]),
                    showlegend=False,
                ))
                if i < len(ins_labels)-1:
                    fig_ins.add_annotation(
                        ax=i, ay=0, x=i+1, y=0,
                        xref="x", yref="y", axref="x", ayref="y",
                        arrowhead=2, arrowcolor="#888888", arrowwidth=1.5,
                    )
            fig_ins.update_layout(
                xaxis=dict(visible=False, range=[-0.5, len(ins_labels)-0.5]),
                yaxis=dict(visible=False, range=[-0.6, 0.4]),
                paper_bgcolor="#0d1117", height=160,
                margin=dict(t=5, b=70, l=5, r=5),
            )
            st.plotly_chart(fig_ins, use_container_width=True)
            st.markdown("---")

        st.markdown(f"""
| | |
|---|---|
| **Plaintiff** | `{meta['src_label']}` |
| **Defendant** | `{meta['dst_label']}` |
| **P(sue)** | **{p_base}** |
| **K-hop** | {meta['k_hop']} |
| **Subgraph** | {meta['n_nodes']} nodes · {meta['n_edges']} edges |
        """)
        st.markdown("---")

        # Storyline cards from JSON
        for s in storyline:
            with st.expander(f"**{s['title']}**", expanded=True):
                st.markdown(s.get(lang_key, s.get("en", "")))

    with col_gauge:
        p_conf = abs(p_base - 0.5) * 2   # 0=uncertain, 1=certain
        entropy = -(p_base * np.log(p_base + 1e-8) +
                    (1-p_base) * np.log(1-p_base + 1e-8)) / np.log(2)  # bits

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(p_base * 100, 1),
            delta={"reference": 50, "suffix": "%"},
            title={"text": "P(sue) (%)"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#ff4b4b" if p_base > 0.7 else "#ffa500"},
                "bgcolor": "#1a1a2e",
                "steps": [
                    {"range": [0,  50],  "color": "#1a3a1a"},
                    {"range": [50, 70],  "color": "#3a3a1a"},
                    {"range": [70, 100], "color": "#3a1a1a"},
                ],
                "threshold": {"line": {"color":"white","width":4},
                              "thickness": 0.75, "value": 50},
            },
        ))
        fig_gauge.update_layout(paper_bgcolor="#0d1117", font_color="white",
                                height=300, margin=dict(t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        c_conf, c_ent = st.columns(2)
        c_conf.metric("Confidence",
                      "High" if p_conf > 0.6 else "Med" if p_conf > 0.3 else "Low",
                      f"{p_conf:.2f}")
        c_ent.metric("Entropy (bits)", f"{entropy:.3f}")

        st.markdown("---")
        st.markdown("#### Top-3 Critical Edges")
        for e in sorted(cf["by_edge"], key=lambda x: x["delta_p"])[:3]:
            st.markdown(
                f"**#{e['rank']}** `{e['edge_type']}`  \n"
                f"ΔP = **{e['delta_p']:+.4f}**  \n"
                f"`{e['src_label'][:16]}` → `{e['dst_label'][:16]}`"
            )
            st.markdown("---")
