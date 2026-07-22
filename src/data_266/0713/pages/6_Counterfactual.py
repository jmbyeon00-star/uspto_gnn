"""Page 6 — Counterfactual: What removal changes the prediction?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from utils.loader import load_graph, load_counterfactual
from utils.plot_utils import delta_bar
from components.graph import build_pyvis
from components.cards import metric_card
from config import PLOTLY_LAYOUT, EDGE_COLORS, TEXT_COLOR

st.set_page_config(page_title="Counterfactual · Patent XAI", layout="wide")

graph = load_graph()
cf    = load_counterfactual()
if not graph or not cf:
    st.error("Run `python compute/run_all.py` first."); st.stop()

nodes  = graph["nodes"]
edges  = graph["edges"]
p_base = cf.get("base_probability", 0.5)
by_type = cf.get("by_type", {})
by_edge = cf.get("by_edge", [])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔄 Counterfactual")
    show_co  = st.checkbox("Company", value=True)
    show_ipc = st.checkbox("IPC",     value=False)
    selected_edge = st.selectbox(
        "Select edge to remove (preview)",
        [f"#{e['rank']} {e['relation']}: {e['source_label'][:14]}→{e['target_label'][:14]}"
         for e in by_edge[:15]],
        index=0,
    )
    st.markdown("---")
    st.metric("Base P(sue)", f"{p_base:.4f}")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔄 Counterfactual — What removal changes the prediction?")
st.markdown(
    "We systematically remove edges and measure how much the litigation probability drops. "
    "**ΔP < 0** means removal hurts the prediction (= that edge is causal)."
)

# ── Relation-level analysis ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("## By Relation Type")

rel_keys   = list(by_type.keys())
rel_deltas = [by_type[k]["delta"]  for k in rel_keys]
rel_afters = [by_type[k]["after"]  for k in rel_keys]

col_bar, col_table = st.columns([3, 2])

with col_bar:
    colors = ["#e74c3c" if d < -0.1 else "#f39c12" if d < -0.02 else "#888"
              for d in rel_deltas]
    fig_rt = go.Figure()
    fig_rt.add_trace(go.Bar(
        x=rel_keys, y=[p_base]*len(rel_keys), name="Before",
        marker_color="#4488ff", opacity=0.4,
    ))
    fig_rt.add_trace(go.Bar(
        x=rel_keys, y=rel_afters, name="After removal",
        marker_color=colors,
        text=[f"{v:.4f}" for v in rel_afters], textposition="outside",
    ))
    fig_rt.add_hline(y=0.5, line_dash="dot", line_color="yellow", opacity=0.5,
                     annotation_text="Decision boundary (0.5)")
    fig_rt.update_layout(
        **PLOTLY_LAYOUT, title="P(sue) Before vs After (by relation)",
        barmode="overlay", height=360, yaxis=dict(range=[0,1.1]),
    )
    st.plotly_chart(fig_rt, use_container_width=True)

with col_table:
    worst = min(by_type, key=lambda k: by_type[k]["delta"])
    metric_card("Most Causal Relation", worst,
                f"Δ = {by_type[worst]['delta']:+.4f}", "#e74c3c", "🎯")
    for rel in rel_keys:
        d = by_type[rel]
        col = "#e74c3c" if d["delta"]<-0.1 else "#f39c12" if d["delta"]<-0.02 else "#2ecc71"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"border-left:3px solid {col};padding:4px 8px;margin:4px 0;"
            f"background:#161b22;border-radius:4px'>"
            f"<b style='color:{TEXT_COLOR}'>{rel}</b>"
            f"<span style='color:{col}'>{d['delta']:+.4f}</span>"
            f"<span style='color:#8b949e'>{d['before']:.3f}→{d['after']:.3f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ── Per-edge analysis ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## By Individual Edge")

col_chart, col_graph = st.columns([1, 2])

with col_chart:
    if by_edge:
        labels = [f"#{e['rank']} {e['relation'][:4]}" for e in by_edge[:15]]
        deltas = [e["delta"] for e in by_edge[:15]]
        colors = ["#e74c3c" if d < -0.05 else "#f39c12" if d < 0 else "#2ecc71"
                  for d in deltas]
        fig_e = go.Figure(go.Bar(
            x=deltas, y=labels, orientation="h",
            marker_color=colors,
            text=[f"{d:+.4f}" for d in deltas], textposition="outside",
        ))
        fig_e.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.3)
        fig_e.update_layout(
            **PLOTLY_LAYOUT, height=450, title="ΔP per removed edge",
            xaxis_title="ΔP(sue)", yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_e, use_container_width=True)

with col_graph:
    st.markdown("**Before** (full graph) vs **After** (selected edge removed)")

    sel_idx = int(selected_edge.split("#")[1].split(" ")[0]) - 1
    if by_edge and sel_idx < len(by_edge):
        sel  = by_edge[sel_idx]
        removed_id = sel.get("edge_id","")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Before**: P = {sel['before']:.4f}")
            html_b = build_pyvis(nodes, edges, height="320px",
                                 show_company=show_co, show_ipc=show_ipc)
            components.html(html_b, height=340)

        with c2:
            st.markdown(f"**After**: P = {sel['after']:.4f}  "
                        f"*(Δ={sel['delta']:+.4f})*")
            edges_after = [e for e in edges if e["id"] != removed_id]
            html_a = build_pyvis(nodes, edges_after, height="320px",
                                 show_company=show_co, show_ipc=show_ipc)
            components.html(html_a, height=340)

        st.markdown(
            f"**Removed**: relation=`{sel['relation']}` · "
            f"`{sel['source_label']}` → `{sel['target_label']}`"
        )

# ── Edge table ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Top Counterfactual Edges")
if by_edge:
    import pandas as pd
    df = pd.DataFrame([{
        "Rank":   e["rank"],
        "Relation": e["relation"],
        "Source": e["source_label"][:20],
        "Target": e["target_label"][:20],
        "Before": f"{e['before']:.4f}",
        "After":  f"{e['after']:.4f}",
        "ΔP":     f"{e['delta']:+.4f}",
        "GNN Imp": f"{e.get('gnn_importance',0):.3f}",
    } for e in by_edge])
    st.dataframe(df, hide_index=True, use_container_width=True)
