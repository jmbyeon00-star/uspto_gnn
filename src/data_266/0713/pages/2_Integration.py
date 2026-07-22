"""Page 2 — Integration: New patent insertion animation."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import streamlit.components.v1 as components
from utils.loader import load_graph, load_prediction
from components.graph import build_pyvis
from components.timeline import integration_diagram
from config import NODE_COLORS

st.set_page_config(page_title="Integration · Patent XAI", layout="wide")

graph = load_graph()
pred  = load_prediction()
if not graph:
    st.error("Run `python compute/run_all.py` first."); st.stop()

nodes = graph["nodes"]
edges = graph["edges"]
meta  = graph["metadata"]

src_pat = meta.get("src_patent", "?")
src_co  = (meta.get("src_owners") or ["Unknown"])[0]
src_ipc = (meta.get("src_ipc") or ["—"])[0]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔗 Integration")
    st.markdown("Animate how a new patent is inserted into the existing graph.")
    st.markdown("---")
    auto_play = st.checkbox("Auto-play", value=False)
    speed     = st.slider("Delay (sec)", 0.5, 3.0, 1.5, 0.5)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔗 Integration — How is a new patent inserted?")
st.markdown(
    "A new patent first exists in isolation. "
    "As we connect it to its **owner company**, **IPC class**, and **cited patents**, "
    "the GNN accumulates evidence for litigation prediction."
)

# ── Animation steps ───────────────────────────────────────────────────────────
STEPS = [
    {"icon": "📄", "label": f"New Patent\n{src_pat}", "color": NODE_COLORS["patent_src"],
     "desc": f"Patent `{src_pat}` is inserted as a new node with no connections.",
     "relations": set()},
    {"icon": "🏢", "label": f"Owner\n{src_co[:14]}", "color": NODE_COLORS["company"],
     "desc": f"Connected to owner company **{src_co}** via an `owner` edge.",
     "relations": {"owner"}},
    {"icon": "🔖", "label": f"IPC\n{src_ipc[:8]}", "color": NODE_COLORS["ipc"],
     "desc": f"Assigned to IPC class `{src_ipc}` — defines the technology domain.",
     "relations": {"owner","ipc"}},
    {"icon": "📎", "label": "Citations", "color": "#f39c12",
     "desc": "Citation edges connect to existing patents in the graph. "
             "These are the most important edges for litigation prediction.",
     "relations": {"owner","ipc","citation"}},
    {"icon": "🕸️", "label": "Full Graph", "color": "#888888",
     "desc": "Fully integrated. The GNN runs message passing over the complete graph "
             "to compute the litigation probability.",
     "relations": {"owner","ipc","citation","litigation"}},
]

# Session state
if "int_step" not in st.session_state:
    st.session_state.int_step = 0
if "int_playing" not in st.session_state:
    st.session_state.int_playing = False

step = st.session_state.int_step

col_ctrl = st.columns([1, 1, 1, 1, 3])
if col_ctrl[0].button("◀ Prev"):
    st.session_state.int_step = max(0, step - 1)
    st.session_state.int_playing = False
if col_ctrl[1].button("Next ▶"):
    st.session_state.int_step = min(len(STEPS)-1, step + 1)
    st.session_state.int_playing = False
play_lbl = "⏸ Pause" if st.session_state.int_playing else "▶ Play"
if col_ctrl[2].button(play_lbl):
    st.session_state.int_playing = not st.session_state.int_playing
if col_ctrl[3].button("↺ Reset"):
    st.session_state.int_step = 0
    st.session_state.int_playing = False

# Progress
st.progress((step + 1) / len(STEPS))
st.markdown(
    f"**Step {step+1}/{len(STEPS)}: {STEPS[step]['icon']} {STEPS[step]['label'].split(chr(10))[0]}**"
    f"  — {STEPS[step]['desc']}"
)

# ── Timeline diagram ──────────────────────────────────────────────────────────
st.plotly_chart(integration_diagram(STEPS, step), use_container_width=True)

# ── Graph for current step ────────────────────────────────────────────────────
visible_relations = STEPS[step]["relations"]
filt_edges = [e for e in edges if e["relation"] in visible_relations]

# At step 0, show only src patent node
if step == 0:
    src_id  = f"pat_{next(n['id'].split('_')[1] for n in nodes if n.get('is_src'))}"
    filt_nodes = [n for n in nodes if n.get("is_src")]
    filt_edges = []
else:
    filt_nodes = nodes
    if "company" not in visible_relations:
        filt_nodes = [n for n in nodes if n["type"] != "company"]
    if "ipc" not in visible_relations:
        filt_nodes = [n for n in nodes if n["type"] != "ipc"]

show_co  = "owner" in visible_relations
show_ipc = "ipc"   in visible_relations
html = build_pyvis(filt_nodes, filt_edges, height="500px",
                   show_company=show_co, show_ipc=show_ipc, threshold=0.0)
components.html(html, height=520, scrolling=False)

# ── Auto-play ─────────────────────────────────────────────────────────────────
if st.session_state.int_playing:
    time.sleep(speed)
    nxt = step + 1
    if nxt >= len(STEPS):
        nxt = 0
        st.session_state.int_playing = False
    st.session_state.int_step = nxt
    st.rerun()
