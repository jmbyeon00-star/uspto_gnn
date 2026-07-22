"""components/legend.py — Node/edge legend for sidebar."""
import streamlit as st
from config import NODE_COLORS, EDGE_COLORS


def render_node_legend():
    st.markdown("**Nodes**")
    items = [
        ("⭐", NODE_COLORS["patent_src"],  "Plaintiff"),
        ("◆", NODE_COLORS["patent_dst"],  "Defendant"),
        ("●", NODE_COLORS["patent"],      "Patent"),
        ("▲", NODE_COLORS["company"],     "Company"),
        ("⬡", NODE_COLORS["ipc"],         "IPC"),
        ("◆", NODE_COLORS["path"],        "Path node"),
    ]
    for sym, color, label in items:
        st.markdown(
            f"<span style='color:{color};font-size:1.1em'>{sym}</span> {label}",
            unsafe_allow_html=True,
        )


def render_edge_legend():
    st.markdown("**Edges**")
    for rel, color in EDGE_COLORS.items():
        st.markdown(
            f"<span style='color:{color}'>■</span> {rel}",
            unsafe_allow_html=True,
        )


def render_scale_legend():
    st.markdown("**Scale**")
    st.caption("Width = Importance / Attention")
    st.caption("Size  = Node importance")
    st.caption("Dashed = low importance")


def render_full_legend():
    render_node_legend()
    st.markdown("---")
    render_edge_legend()
    st.markdown("---")
    render_scale_legend()
