"""utils/plot_utils.py — Plotly helpers for consistent dark-theme charts."""
import plotly.graph_objects as go
import numpy as np
from config import PLOTLY_LAYOUT, EDGE_COLORS, BG_COLOR, TEXT_COLOR


def bar_chart(labels, values, colors=None, title="", xlabel="", ylabel="",
              height=320, text_vals=None) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors or "#4488ff",
        text=text_vals or [f"{v:.3f}" for v in values],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, title=title, height=height,
        xaxis_title=xlabel, yaxis_title=ylabel,
    )
    return fig


def scatter_chart(x, y, labels=None, colors=None, sizes=None,
                  title="", xlabel="", ylabel="", height=350) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=x, y=y,
        mode="markers+text" if labels else "markers",
        text=labels,
        textposition="top center",
        textfont=dict(size=8, color=TEXT_COLOR),
        marker=dict(
            size=sizes or 10,
            color=colors or "#4488ff",
            line=dict(color="white", width=0.8),
            opacity=0.85,
        ),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=height,
                      xaxis_title=xlabel, yaxis_title=ylabel)
    return fig


def probability_bar(pred: dict, height=200) -> go.Figure:
    """Horizontal probability gauge-bar."""
    p = pred.get("probability", 0.0)
    color = "#e74c3c" if p > 0.7 else "#f39c12" if p > 0.5 else "#2ecc71"
    fig = go.Figure(go.Bar(
        x=[p], y=["P(sue)"],
        orientation="h",
        marker_color=color,
        text=[f"{p:.4f}"],
        textposition="inside",
        textfont=dict(size=18, color="white"),
    ))
    fig.add_vline(x=pred.get("threshold", 0.5), line_dash="dash",
                  line_color="white", opacity=0.6,
                  annotation_text=f"thr={pred.get('threshold',0.5):.2f}",
                  annotation_font_color="white")
    fig.update_layout(**PLOTLY_LAYOUT, height=height,
                      xaxis=dict(range=[0, 1], title="Probability"),
                      yaxis=dict(visible=False))
    return fig


def delta_bar(items: list, key_label: str, key_delta: str,
              title="", height=350) -> go.Figure:
    labels = [str(it[key_label]) for it in items]
    deltas = [it[key_delta] for it in items]
    colors = ["#e74c3c" if d < -0.1 else "#f39c12" if d < 0 else "#2ecc71"
              for d in deltas]
    fig = go.Figure(go.Bar(
        x=labels, y=deltas, marker_color=colors,
        text=[f"{d:+.3f}" for d in deltas], textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=height,
                      yaxis_title="ΔP")
    return fig


def attention_heatmap(layer_data: list, title="", height=300) -> go.Figure:
    """Bar chart showing edge attention values for a single layer."""
    eids   = [r["edge_id"]   for r in layer_data if r["attention"] > 0]
    vals   = [r["attention"] for r in layer_data if r["attention"] > 0]
    if not eids:
        return go.Figure().update_layout(**PLOTLY_LAYOUT, title=title+" (no data)", height=height)
    sorted_pairs = sorted(zip(vals, eids), reverse=True)[:20]
    vals_s, eids_s = zip(*sorted_pairs)
    colors = [f"rgb({int(40+200*v)},{int(60+100*v)},{int(180+60*v)})"
              for v in vals_s]
    fig = go.Figure(go.Bar(
        x=list(eids_s), y=list(vals_s), marker_color=colors,
        text=[f"{v:.3f}" for v in vals_s], textposition="outside",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=height,
                      xaxis_title="Edge ID", yaxis_title="Attention")
    return fig
