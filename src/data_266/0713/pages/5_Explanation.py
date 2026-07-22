"""Page 5 — Explanation: GNNExplainer | Attention | PGExplainer | SubgraphX."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from utils.loader import load_graph, load_explanation, load_attention
from utils.plot_utils import bar_chart
from utils.graph_utils import top_k_edges
from components.graph import build_pyvis
from components.legend import render_full_legend
from config import PLOTLY_LAYOUT, TEXT_COLOR, NODE_COLORS

st.set_page_config(page_title="Explanation · Patent XAI", layout="wide")

graph = load_graph()
expln = load_explanation()
attn_j = load_attention()
if not graph:
    st.error("Run `python compute/run_all.py` first."); st.stop()

nodes = graph["nodes"]
edges = graph["edges"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Explanation")
    top_k  = st.slider("Top-K edges to highlight", 3, 15, 8)
    thresh = st.slider("Min importance threshold",  0.0, 1.0, 0.0, 0.01)
    show_co  = st.checkbox("Company", value=True)
    show_ipc = st.checkbox("IPC",     value=False)
    st.markdown("---")
    render_full_legend()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔍 Explanation — Why was this predicted?")
st.markdown(
    "Four XAI methods explain the same prediction from different angles. "
    "All use the **same graph** but differ in *how* they attribute importance."
)

# ── 4-method summary row ─────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.markdown("**GNNExplainer**\nOptimizes edge mask to maximize prediction")
c2.markdown("**Attention**\nNative GAT attention weights (intrinsic)")
c3.markdown("**PGExplainer**\nParametric, generalizes across instances")
c4.markdown("**SubgraphX**\nGreedy coalition: minimal node subset")

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_gnn, tab_attn, tab_pg, tab_sgx, tab_overlay = st.tabs([
    "GNNExplainer", "🧠 Attention", "PGExplainer", "SubgraphX", "⊕ Overlay"
])

# ════════════════════════════════════════════════════════════════════════════
# GNNExplainer
# ════════════════════════════════════════════════════════════════════════════
with tab_gnn:
    st.markdown("### GNNExplainer — Edge Mask Importance")
    st.markdown(
        "GNNExplainer optimizes a **continuous edge mask** that, when applied, "
        "preserves the model's prediction. Edges with high mask value are the "
        "most important for the prediction."
    )

    gnn  = expln.get("gnn_explainer", {})
    edge_imp = gnn.get("edge_importance", {})
    node_imp = gnn.get("node_importance", {})

    col_g, col_bar = st.columns([3, 1])

    tk_ids = set(sorted(edge_imp, key=lambda k: -edge_imp.get(k,0))[:top_k])
    with col_g:
        html = build_pyvis(nodes, edges, height="520px", mode="importance",
                           threshold=thresh, top_k_ids=tk_ids,
                           show_company=show_co, show_ipc=show_ipc)
        components.html(html, height=540)

    with col_bar:
        if edge_imp:
            top = sorted(edge_imp.items(), key=lambda x:-x[1])[:top_k]
            eids, vals = zip(*top)
            colors = [f"rgb({int(40+200*v)},{30},{30})" for v in vals]
            fig = go.Figure(go.Bar(
                y=list(eids), x=list(vals), orientation="h",
                marker_color=colors,
                text=[f"{v:.3f}" for v in vals], textposition="outside",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=500,
                              title=f"Top-{top_k} Edges",
                              xaxis=dict(range=[0,1.1]),
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    # Node importance table
    if node_imp:
        st.markdown("**Node Importance (GNNExplainer)**")
        node_by = {n["id"]: n for n in nodes}
        ni_sorted = sorted(node_imp.items(), key=lambda x:-x[1])[:10]
        rows = [{"Node ID": nid,
                 "Label": node_by.get(nid,{}).get("label","?")[:20],
                 "Type":  node_by.get(nid,{}).get("type","?"),
                 "Importance": f"{v:.4f}"}
                for nid, v in ni_sorted]
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# Attention
# ════════════════════════════════════════════════════════════════════════════
with tab_attn:
    st.markdown("### Native GAT Attention Weights")
    st.markdown(
        "These weights are **intrinsic** to the model — extracted directly from "
        "`GATConv(return_attention_weights=True)`. No post-hoc approximation."
    )

    layer_sel = st.radio("Layer", ["layer1","layer2","layer3","layer4"],
                         format_func=lambda l: l.replace("layer","Layer "),
                         horizontal=True)

    col_g2, col_bar2 = st.columns([3, 1])

    # Get top-k by attention at selected layer
    attn_tk = set()
    for e in edges:
        attn_tk.add(e["id"])  # all visible

    with col_g2:
        html2 = build_pyvis(nodes, edges, height="520px",
                            mode=f"attn_{layer_sel}",
                            attn_layer=layer_sel,
                            threshold=thresh,
                            show_company=show_co, show_ipc=show_ipc)
        components.html(html2, height=540)

    with col_bar2:
        layer_recs = attn_j.get("layers", {}).get(layer_sel, [])
        if layer_recs:
            top_a = sorted(layer_recs, key=lambda r: -r["attention"])[:top_k]
            eids  = [r["edge_id"] for r in top_a]
            vals  = [r["attention"] for r in top_a]
            colors = [f"rgb({30},{80},{int(60+195*v)})" for v in vals]
            fig2 = go.Figure(go.Bar(
                y=eids, x=vals, orientation="h",
                marker_color=colors,
                text=[f"{v:.3f}" for v in vals], textposition="outside",
            ))
            fig2.update_layout(**PLOTLY_LAYOUT, height=500,
                               title=f"Top-{top_k} by Attention",
                               xaxis=dict(range=[0,1.1]),
                               yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig2, use_container_width=True)

    # Layer comparison chart
    st.markdown("**Average attention per layer (top-K edges)**")
    layer_avg = attn_j.get("layer_avg", {})
    if layer_avg:
        fig_la = bar_chart(
            [k.replace("layer","L") for k in layer_avg],
            list(layer_avg.values()),
            colors=["#4488ff"]*4,
            title="Avg. Attention per Layer",
            height=200,
        )
        st.plotly_chart(fig_la, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PGExplainer
# ════════════════════════════════════════════════════════════════════════════
with tab_pg:
    st.markdown("### PGExplainer — Parametric Edge Mask")
    st.markdown(
        "PGExplainer trains a **separate neural network** to predict edge masks. "
        "Unlike GNNExplainer (per-instance), PGExplainer generalizes across the graph. "
        "Edge width = PGExplainer importance."
    )

    pg   = expln.get("pg_explainer", {})
    pg_imp = pg.get("edge_importance", {})

    if not pg_imp:
        st.warning("PGExplainer results not available (see explanation.json). "
                   "Check if PGExplainer training succeeded in 03_explanation.py.")
        # Fallback: show GNNExplainer
        pg_imp = expln.get("gnn_explainer", {}).get("edge_importance", {})
        st.info("Showing GNNExplainer as fallback.")

    col_g3, col_bar3 = st.columns([3, 1])
    pg_tk = set(sorted(pg_imp, key=lambda k: -pg_imp.get(k,0))[:top_k])

    # Patch edge importance with PG values for display
    pg_edges = []
    for e in edges:
        ec = dict(e)
        ec["importance"] = pg_imp.get(e["id"], 0.0)
        pg_edges.append(ec)

    with col_g3:
        html3 = build_pyvis(nodes, pg_edges, height="520px", mode="importance",
                            threshold=thresh, top_k_ids=pg_tk,
                            show_company=show_co, show_ipc=show_ipc)
        components.html(html3, height=540)

    with col_bar3:
        if pg_imp:
            top_p = sorted(pg_imp.items(), key=lambda x:-x[1])[:top_k]
            fig3 = go.Figure(go.Bar(
                y=[x[0] for x in top_p], x=[x[1] for x in top_p],
                orientation="h",
                marker_color="#2ecc71",
                text=[f"{x[1]:.3f}" for x in top_p], textposition="outside",
            ))
            fig3.update_layout(**PLOTLY_LAYOUT, height=500,
                               title=f"Top-{top_k} (PG)",
                               xaxis=dict(range=[0,1.1]),
                               yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# SubgraphX
# ════════════════════════════════════════════════════════════════════════════
with tab_sgx:
    st.markdown("### SubgraphX — Minimal Coalition")
    st.markdown(
        "SubgraphX finds the **smallest subset of nodes** (coalition) "
        "that maintains the prediction probability. "
        "Node size = importance (how much the prediction drops if removed)."
    )

    sgx  = expln.get("subgraph_x", {})
    sgx_node_imp = sgx.get("node_importance", {})
    coalition    = sgx.get("coalition", [])

    col_g4, col_info4 = st.columns([3, 1])

    # Patch node importance for display
    sgx_nodes = []
    for n in nodes:
        nc = dict(n)
        nc["importance"] = max(0.1, sgx_node_imp.get(n["id"], 0.0) * 5)  # amplify
        sgx_nodes.append(nc)

    with col_g4:
        html4 = build_pyvis(sgx_nodes, edges, height="520px", mode="importance",
                            highlight_nodes=set(coalition),
                            threshold=thresh, show_company=show_co, show_ipc=show_ipc)
        components.html(html4, height=540)

    with col_info4:
        st.markdown("**Coalition (top nodes)**")
        node_by_id = {n["id"]: n for n in nodes}
        for nid in coalition:
            n = node_by_id.get(nid, {})
            imp = sgx_node_imp.get(nid, 0.0)
            st.markdown(
                f"🔶 `{n.get('label','?')[:16]}`  \n"
                f"type={n.get('type','?')}  imp={imp:.4f}"
            )
        st.markdown("---")
        st.caption(
            "Nodes in the coalition are the minimal set "
            "that preserves the litigation prediction."
        )

        # Importance bar
        if sgx_node_imp:
            top_s = sorted(sgx_node_imp.items(), key=lambda x:-x[1])[:8]
            fig4 = go.Figure(go.Bar(
                y=[node_by_id.get(k,{}).get("label","?")[:14] for k,_ in top_s],
                x=[v for _,v in top_s],
                orientation="h",
                marker_color="#f39c12",
                text=[f"{v:.4f}" for _,v in top_s], textposition="outside",
            ))
            fig4.update_layout(**PLOTLY_LAYOUT, height=300,
                               title="Node LOO Importance",
                               yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# Overlay (all 3 methods)
# ════════════════════════════════════════════════════════════════════════════
with tab_overlay:
    st.markdown("### ⊕ Three-Method Overlay")
    st.markdown("""
| Color Channel | Method | Meaning |
|---|---|---|
| 🔴 Red    | GNNExplainer    | Post-hoc optimized mask |
| 🔵 Blue   | Attention (L2)  | Intrinsic model weights |
| 🟢 Green  | Counterfactual  | Causal impact |
> **White / bright** = all three methods agree this edge is important.
""")

    col_g5, col_leg5 = st.columns([4, 1])
    with col_g5:
        html5 = build_pyvis(nodes, edges, height="540px", mode="overlay",
                            show_company=show_co, show_ipc=show_ipc, threshold=thresh)
        components.html(html5, height=560)

    with col_leg5:
        st.markdown("**Method colors**")
        for lbl, col in [("GNNExplainer","#e74c3c"),
                          ("Attention L2","#4488ff"),
                          ("Counterfactual","#2ecc71"),
                          ("All agree","#ffffff")]:
            st.markdown(f"<span style='color:{col}'>■■■</span> {lbl}",
                        unsafe_allow_html=True)
        st.markdown("---")

        # Scatter: GNN vs Attention
        import numpy as np
        gnn_imp  = expln.get("gnn_explainer",{}).get("edge_importance",{})
        attn_l2  = {r["edge_id"]: r["attention"]
                    for r in attn_j.get("layers",{}).get("layer2",[])}
        shared   = list(gnn_imp.keys() & attn_l2.keys())
        if shared:
            x_vals = [gnn_imp[k] for k in shared]
            y_vals = [attn_l2[k] for k in shared]
            try:
                from scipy.stats import pearsonr, spearmanr
                r_p, _ = pearsonr(x_vals, y_vals)
                r_s, _ = spearmanr(x_vals, y_vals)
                st.metric("Pearson r",  f"{r_p:.3f}")
                st.metric("Spearman ρ", f"{r_s:.3f}")
                if abs(r_p) > 0.6:
                    st.success("Strong agreement")
                elif abs(r_p) > 0.3:
                    st.warning("Moderate agreement")
                else:
                    st.error("Weak agreement")
            except ImportError:
                st.caption("scipy not installed")
