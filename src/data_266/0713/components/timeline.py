"""components/timeline.py — Conclusion timeline + integration animation."""
import plotly.graph_objects as go
import streamlit as st
from config import BG_COLOR, TEXT_COLOR


def conclusion_timeline(cards: list) -> go.Figure:
    """Horizontal timeline connecting storyline cards."""
    n = len(cards)
    if n == 0:
        return go.Figure()

    x  = list(range(n))
    colors_map = {1: "#4488ff", 2: "#2ecc71", 3: "#f39c12", 4: "#e74c3c"}
    colors = [colors_map.get(c.get("step", i+1), "#888") for i, c in enumerate(cards)]
    icons  = [c.get("icon", "") for c in cards]
    vals   = [str(c.get("value") or "") for c in cards]
    titles = [c.get("title","") for c in cards]

    fig = go.Figure()

    # Connecting line
    fig.add_trace(go.Scatter(
        x=x, y=[0]*n, mode="lines",
        line=dict(color="#30363d", width=3), showlegend=False,
    ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=x, y=[0]*n, mode="markers+text",
        marker=dict(size=44, color=colors, line=dict(color="white", width=2),
                    symbol="circle"),
        text=[f"{ic}\n{vl}" for ic, vl in zip(icons, vals)],
        textfont=dict(size=11, color="white"),
        textposition="middle center",
        showlegend=False,
        hovertemplate=[f"<b>{t}</b>" for t in titles],
    ))

    # Labels below
    for i, c in enumerate(cards):
        fig.add_annotation(
            x=i, y=-0.28,
            text=f"<b>{c.get('title','')}</b>",
            showarrow=False,
            font=dict(size=12, color=TEXT_COLOR),
        )

    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        xaxis=dict(visible=False, range=[-0.5, n-0.5]),
        yaxis=dict(visible=False, range=[-0.5, 0.4]),
        height=200,
        margin=dict(t=10, b=60, l=20, r=20),
    )
    return fig


def integration_diagram(steps: list, step_idx: int) -> go.Figure:
    """Step-by-step new patent integration animation."""
    visible = steps[:step_idx + 1]
    n       = len(visible)

    icons   = [s.get("icon", "●")  for s in visible]
    labels  = [s.get("label", "?") for s in visible]
    colors  = [s.get("color", "#888") for s in visible]

    fig = go.Figure()

    if n > 1:
        fig.add_trace(go.Scatter(
            x=list(range(n)), y=[0]*n, mode="lines",
            line=dict(color="#30363d", width=2), showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=list(range(n)), y=[0]*n,
        mode="markers+text",
        marker=dict(size=40, color=colors, line=dict(color="white", width=2)),
        text=icons, textposition="middle center",
        textfont=dict(size=18, color="white"),
        showlegend=False,
    ))
    for i, lbl in enumerate(labels):
        fig.add_annotation(
            x=i, y=-0.25, text=lbl,
            showarrow=False, font=dict(size=10, color=TEXT_COLOR),
        )

    fig.update_layout(
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        xaxis=dict(visible=False, range=[-0.5, len(steps)-0.5]),
        yaxis=dict(visible=False, range=[-0.5, 0.35]),
        height=180, margin=dict(t=5, b=60, l=10, r=10),
    )
    return fig
