"""Page 1 — Overview: Full heterograph + statistics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from utils.loader import load_graph, load_prediction
from components.graph import build_pyvis
from components.legend import render_full_legend
from config import BG_COLOR, TEXT_COLOR, EDGE_COLORS, NODE_COLORS, PLOTLY_LAYOUT

st.set_page_config(page_title="Overview · Patent XAI", layout="wide")

graph = load_graph()
pred  = load_prediction()
if not graph:
    st.error("Run `python compute/run_all.py` first."); st.stop()

nodes = graph["nodes"]
edges = graph["edges"]
meta  = graph["metadata"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌐 Overview")
    show_company = st.checkbox("Company nodes", value=True)
    show_ipc     = st.checkbox("IPC nodes",     value=False)
    show_relations = st.multiselect(
        "Show relations",
        ["citation","litigation","owner","ipc"],
        default=["citation","litigation","owner"],
    )
    st.markdown("---")
    render_full_legend()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🌐 Overview — What did the GNN learn?")
st.markdown(
    "The GNN was trained on this **Heterogeneous Patent Graph** "
    "containing citations, ownership, IPC codes, and litigation links. "
    "The query pair is highlighted: ⭐ Plaintiff · ◆ Defendant."
)

# ── Metrics ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
n_types = {n["type"] for n in nodes}
c1.metric("Total Nodes",    meta.get("node_count", len(nodes)))
c2.metric("Total Edges",    meta.get("edge_count", len(edges)))
c3.metric("Patent nodes",   sum(1 for n in nodes if n["type"]=="patent"))
c4.metric("Company nodes",  sum(1 for n in nodes if n["type"]=="company"))
c5.metric("IPC nodes",      sum(1 for n in nodes if n["type"]=="ipc"))

st.markdown("---")

# ── Graph ─────────────────────────────────────────────────────────────────────
col_g, col_info = st.columns([4, 1])

filt_edges = [e for e in edges if e["relation"] in show_relations]
html = build_pyvis(nodes, filt_edges, height="600px",
                   show_company=show_company, show_ipc=show_ipc,
                   threshold=0.0)

with col_g:
    components.html(html, height=630, scrolling=False)

with col_info:
    st.markdown("**Pair Info**")
    p = pred.get("probability", 0.0)
    pc = "#e74c3c" if p>0.7 else "#f39c12" if p>0.5 else "#2ecc71"
    st.markdown(f"⭐ `{meta.get('src_patent','?')}`")
    st.markdown(f"Plaintiff: {', '.join(meta.get('src_owners',[]) or ['—'])}")
    st.markdown(f"IPC: `{', '.join(meta.get('src_ipc',[]) or ['—'])}`")
    st.markdown("---")
    st.markdown(f"◆ `{meta.get('dst_patent','?')}`")
    st.markdown(f"Defendant: {', '.join(meta.get('dst_owners',[]) or ['—'])}")
    st.markdown(f"IPC: `{', '.join(meta.get('dst_ipc',[]) or ['—'])}`")
    st.markdown("---")
    st.markdown(
        f"**P(sue)** = <span style='color:{pc};font-size:1.3em;font-weight:700'>"
        f"{p:.4f}</span>", unsafe_allow_html=True
    )
    st.markdown("---")
    render_full_legend()

# ── Statistics ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Graph Statistics")
col_nt, col_et = st.columns(2)

with col_nt:
    type_counts = {}
    for n in nodes:
        type_counts[n["type"]] = type_counts.get(n["type"], 0) + 1
    fig_nt = go.Figure(go.Pie(
        labels=list(type_counts.keys()),
        values=list(type_counts.values()),
        marker_colors=[NODE_COLORS.get(t, "#888") for t in type_counts],
        hole=0.4,
    ))
    fig_nt.update_layout(**PLOTLY_LAYOUT, title="Node Type Distribution", height=280)
    st.plotly_chart(fig_nt, use_container_width=True)

with col_et:
    rel_counts = {}
    for e in edges:
        rel_counts[e["relation"]] = rel_counts.get(e["relation"], 0) + 1
    fig_et = go.Figure(go.Bar(
        x=list(rel_counts.keys()),
        y=list(rel_counts.values()),
        marker_color=[EDGE_COLORS.get(r, "#888") for r in rel_counts],
        text=list(rel_counts.values()), textposition="outside",
    ))
    fig_et.update_layout(**PLOTLY_LAYOUT, title="Edge Relation Distribution",
                         height=280, yaxis_title="Count")
    st.plotly_chart(fig_et, use_container_width=True)
