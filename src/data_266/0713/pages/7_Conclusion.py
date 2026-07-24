"""Page 7 — Conclusion: Final storyline, timeline, evidence path."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from utils.loader import (load_storyline, load_prediction, load_evidence_path,
                           load_graph, load_counterfactual)
from components.timeline import conclusion_timeline
from components.cards import story_card, info_banner
from components.gauge import probability_gauge
from components.graph import build_pyvis
from config import PLOTLY_LAYOUT, TEXT_COLOR, BG_COLOR, NODE_COLORS

st.set_page_config(page_title="Conclusion · Patent XAI", layout="wide")

story = load_storyline()
pred  = load_prediction()
ep    = load_evidence_path()
graph = load_graph()
cf    = load_counterfactual()

if not story:
    st.error("Run `python compute/run_all.py` first."); st.stop()

nodes = graph.get("nodes", []) if graph else []
edges = graph.get("edges", []) if graph else []
p     = pred.get("probability", 0.0)
cards = story.get("cards", [])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✅ Conclusion")
    st.metric("P(sue)", f"{p:.4f}")
    st.metric("Prediction", pred.get("prediction","?").upper())
    st.metric("Confidence",  pred.get("confidence","?").upper())

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ✅ Conclusion — Final Explanation")
pc = "#e74c3c" if p>0.7 else "#f39c12" if p>0.5 else "#2ecc71"
st.markdown(
    f"### "
    f"<span style='color:{pc}'>"
    f"{'⚖️ Litigation PREDICTED' if pred.get('prediction')=='sue' else '✅ No Litigation Predicted'}"
    f"</span> — P(sue) = **{p:.4f}**",
    unsafe_allow_html=True,
)
info_banner(story.get("summary",""))
st.markdown("---")

# ── Timeline ──────────────────────────────────────────────────────────────────
st.markdown("### Explanation Flow")
if cards:
    st.plotly_chart(conclusion_timeline(cards), use_container_width=True)

# ── Story cards ───────────────────────────────────────────────────────────────
st.markdown("### Step-by-step Narrative")
for card in cards:
    story_card(card)

st.markdown("---")

# ── Evidence path ─────────────────────────────────────────────────────────────
col_path, col_gauge = st.columns([3, 1])

with col_path:
    st.markdown("### Evidence Path")
    path_nodes = ep.get("path", [])
    path_edges = ep.get("edges", [])
    path_score = ep.get("score", 0.0)
    path_hops  = ep.get("hops", 0)
    path_labels = ep.get("path_labels", path_nodes)

    if path_nodes:
        st.caption(f"Path length: {path_hops} hops · Score: {path_score:.4f}")

        n = len(path_nodes)
        fig_flow = go.Figure()
        # Connect line
        fig_flow.add_trace(go.Scatter(
            x=list(range(n)), y=[0]*n, mode="lines",
            line=dict(color="#30363d", width=3), showlegend=False,
        ))
        # Node markers
        node_map = {n_["id"]: n_ for n_ in nodes}
        for i, (nid, lbl) in enumerate(zip(path_nodes, path_labels)):
            n_  = node_map.get(nid, {})
            is_src = n_.get("is_src", False)
            is_dst = n_.get("is_dst", False)
            color  = (NODE_COLORS["patent_src"] if is_src else
                      NODE_COLORS["patent_dst"] if is_dst else
                      NODE_COLORS.get(n_.get("type","patent"), "#888"))
            icon   = "⭐" if is_src else "◆" if is_dst else "●"
            fig_flow.add_trace(go.Scatter(
                x=[i], y=[0], mode="markers+text",
                marker=dict(size=40, color=color, line=dict(color="white",width=2)),
                text=[icon], textfont=dict(size=16, color="white"),
                textposition="middle center",
                showlegend=False,
                hovertemplate=f"<b>{lbl}</b><extra></extra>",
            ))
            fig_flow.add_annotation(
                x=i, y=-0.22, text=str(lbl)[:16],
                showarrow=False, font=dict(size=9, color=TEXT_COLOR),
            )
            # Edge label
            if i < len(path_edges):
                pe = path_edges[i]
                fig_flow.add_annotation(
                    x=i+0.5, y=0.15,
                    text=f"{pe.get('relation','')}\n{pe.get('importance',0):.2f}",
                    showarrow=False, font=dict(size=8, color="#8b949e"),
                )
                fig_flow.add_annotation(
                    ax=i, ay=0, x=i+1, y=0,
                    xref="x", yref="y", axref="x", ayref="y",
                    arrowhead=2, arrowcolor="#f1c40f", arrowwidth=2,
                )

        fig_flow.update_layout(
            paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
            xaxis=dict(visible=False, range=[-0.5, n-0.5]),
            yaxis=dict(visible=False, range=[-0.4, 0.3]),
            height=200, margin=dict(t=10, b=70, l=10, r=10),
        )
        st.plotly_chart(fig_flow, use_container_width=True)

        # Path graph
        path_set   = set(path_nodes)
        path_pairs = {(pe["source"], pe["target"]) for pe in path_edges}
        html_path = build_pyvis(nodes, edges, height="360px",
                                path_node_ids=path_set,
                                path_edge_pairs=path_pairs)
        components.html(html_path, height=380)
    else:
        st.warning("Evidence path not found.")

with col_gauge:
    st.markdown("### Prediction")
    st.plotly_chart(probability_gauge(p, pred.get("threshold", 0.5), height=280),
                    use_container_width=True)

    st.markdown("---")
    st.markdown("### Key Numbers")
    cf_by_type = cf.get("by_type", {}) if cf else {}
    if cf_by_type:
        worst = min(cf_by_type, key=lambda k: cf_by_type[k]["delta"])
        st.metric("Most causal", worst,
                  f"Δ={cf_by_type[worst]['delta']:+.4f}")

    st.metric("Evidence path hops",  path_hops)
    st.metric("Evidence path score", f"{path_score:.4f}")

# ── Counterfactual summary ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Counterfactual Summary")
st.markdown(story.get("counterfactual_result","—"))

col_c1, col_c2 = st.columns(2)
with col_c1:
    st.markdown("**Main reason**")
    st.info(story.get("main_reason","—"))
with col_c2:
    st.markdown("**Secondary reason**")
    st.info(story.get("secondary_reason","—") or "*No secondary reason identified.*")

# ── Final conclusion box ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"""<div style="background:#161b22;border:1px solid #30363d;
        border-left:6px solid {pc};border-radius:12px;
        padding:20px 24px;margin-top:8px;">
      <div style="font-size:0.8em;color:#8b949e;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:8px;">FINAL CONCLUSION</div>
      <div style="font-size:1.05em;color:#e6edf3;line-height:1.7;">
        {story.get("final_conclusion","—")}
      </div>
    </div>""",
    unsafe_allow_html=True,
)
