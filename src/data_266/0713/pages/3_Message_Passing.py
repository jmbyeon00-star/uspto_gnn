"""Page 3 — Message Passing: Hop slider + Layer selector with attention."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import streamlit.components.v1 as components
from utils.loader import load_graph, load_attention
from utils.plot_utils import attention_heatmap
from components.graph import build_pyvis
from components.legend import render_full_legend

st.set_page_config(page_title="Message Passing · Patent XAI", layout="wide")

graph  = load_graph()
attn_j = load_attention()
if not graph:
    st.error("Run `python compute/run_all.py` first."); st.stop()

nodes = graph["nodes"]
edges = graph["edges"]

LAYER_NAMES = ["layer1", "layer2", "layer3", "layer4"]
LAYER_LABELS = {
    "layer1": "Layer 1 (×8 heads) — Initial neighbourhood",
    "layer2": "Layer 2 (×4 heads) — 1-hop aggregation",
    "layer3": "Layer 3 (×4 heads) — 2-hop aggregation",
    "layer4": "Layer 4 (×2 heads) — Global context",
}
HOP_DESCS = [
    "Hop 0 — Only the query pair (Plaintiff ⭐, Defendant ◆)",
    "Hop 1 — Direct neighbours: patents cited by / citing the pair",
    "Hop 2 — Second-order neighbours: extended citation context",
    "Hop 3 — Full receptive field of the GNN",
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Message Passing")
    hop   = st.slider("Hop radius", 0, 3, 1)
    layer = st.radio("Attention layer", LAYER_NAMES,
                     format_func=lambda k: LAYER_LABELS[k].split(" — ")[0],
                     horizontal=False)
    show_company = st.checkbox("Show Company", value=True)
    show_ipc     = st.checkbox("Show IPC",     value=False)
    animate      = st.checkbox("Auto-animate hops", value=False)
    speed        = st.slider("Delay (sec)", 0.5, 2.5, 1.2, 0.5)
    st.markdown("---")
    render_full_legend()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📡 Message Passing — How does information propagate?")
st.markdown(
    "Each GNN layer aggregates information from **one additional hop**. "
    "Edge width = attention weight at the selected layer. "
    "Drag the **Hop slider** to expand the receptive field."
)

# ── Controls ──────────────────────────────────────────────────────────────────
col_h, col_l = st.columns(2)
with col_h:
    st.info(f"**{HOP_DESCS[hop]}**")
with col_l:
    st.info(f"**{LAYER_LABELS[layer]}**")

# ── Layer avg attention metrics ───────────────────────────────────────────────
layer_avg = attn_j.get("layer_avg", {})
if layer_avg:
    cols = st.columns(4)
    for i, lk in enumerate(LAYER_NAMES):
        v = layer_avg.get(lk, 0.0)
        cols[i].metric(f"L{i+1} avg attn", f"{v:.4f}",
                       "↑ highest" if lk == max(layer_avg, key=layer_avg.get) else "")

st.markdown("---")

# ── Graph ─────────────────────────────────────────────────────────────────────
col_g, col_chart = st.columns([3, 1])

with col_g:
    html = build_pyvis(
        nodes, edges,
        height="560px",
        mode=f"attn_{layer}",
        attn_layer=layer,
        hop_limit=hop,
        show_company=show_company,
        show_ipc=show_ipc,
        threshold=0.0,
    )
    components.html(html, height=580, scrolling=False)

    # Hop progress bar
    st.progress((hop + 1) / 4)
    n_visible = sum(
        1 for n in nodes
        if n.get("min_hop") is None or n["min_hop"] <= hop
    )
    st.caption(f"Hop {hop}: {n_visible} nodes visible in receptive field")

with col_chart:
    st.markdown(f"**{LAYER_LABELS[layer].split(' — ')[0]} Attention Distribution**")
    layer_recs = attn_j.get("layers", {}).get(layer, [])
    if layer_recs:
        fig = attention_heatmap(layer_recs, height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No attention data.")

# ── All-layer comparison ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Attention across all layers")
cols = st.columns(4)
for i, lk in enumerate(LAYER_NAMES):
    recs = attn_j.get("layers", {}).get(lk, [])
    with cols[i]:
        st.markdown(f"**{LAYER_LABELS[lk].split(' ')[0]} {LAYER_LABELS[lk].split(' ')[1]}**")
        if recs:
            top5 = sorted(recs, key=lambda r: -r["attention"])[:5]
            for r in top5:
                w = int(r["normalized_attention"] * 100)
                st.markdown(
                    f"`{r['edge_id']}` "
                    f"<div style='height:4px;width:{w}%;background:#4488ff;"
                    f"border-radius:2px;margin:2px 0'></div>"
                    f"{r['attention']:.3f}",
                    unsafe_allow_html=True,
                )

# ── Auto-animate ──────────────────────────────────────────────────────────────
if animate and "mp_hop" not in st.session_state:
    st.session_state.mp_hop = 0

if animate:
    if "mp_hop" not in st.session_state:
        st.session_state.mp_hop = 0
    time.sleep(speed)
    st.session_state.mp_hop = (st.session_state.mp_hop + 1) % 4
    st.rerun()
