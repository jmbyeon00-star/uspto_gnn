"""components/gauge.py — Probability gauge chart."""
import plotly.graph_objects as go
import numpy as np
from config import BG_COLOR, TEXT_COLOR


def probability_gauge(probability: float, threshold: float = 0.5,
                      title: str = "P(sue)", height: int = 320) -> go.Figure:
    p     = float(probability)
    color = "#e74c3c" if p > 0.7 else "#f39c12" if p > 0.5 else "#2ecc71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(p * 100, 2),
        delta={"reference": threshold * 100, "suffix": "%",
               "increasing": {"color": "#e74c3c"},
               "decreasing": {"color": "#2ecc71"}},
        title={"text": title, "font": {"size": 18, "color": TEXT_COLOR}},
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": TEXT_COLOR},
            "bar":   {"color": color, "thickness": 0.28},
            "bgcolor": "#161b22",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0,  50],  "color": "#1a3a1a"},
                {"range": [50, 70],  "color": "#3a3a1a"},
                {"range": [70, 100], "color": "#3a1a1a"},
            ],
            "threshold": {
                "line":      {"color": "white", "width": 4},
                "thickness": 0.75,
                "value":     threshold * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor=BG_COLOR, font_color=TEXT_COLOR,
        height=height, margin=dict(t=50, b=10, l=20, r=20),
    )
    return fig


def confidence_bar(probability: float, height: int = 80) -> go.Figure:
    p      = float(probability)
    color  = "#e74c3c" if p > 0.7 else "#f39c12" if p > 0.5 else "#2ecc71"
    label  = "HIGH" if (p > 0.7 or p < 0.3) else "MEDIUM" if (p > 0.6 or p < 0.4) else "LOW"
    conf   = abs(p - 0.5) * 2
    entropy = -(p * np.log(p + 1e-9) + (1-p) * np.log(1-p + 1e-9)) / np.log(2)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[conf], y=["Confidence"], orientation="h",
        marker_color=color,
        text=[f"{label}  (entropy={entropy:.3f} bits)"],
        textposition="inside", textfont=dict(size=14, color="white"),
    ))
    fig.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        xaxis=dict(range=[0, 1], visible=False),
        yaxis=dict(visible=False),
        height=height, margin=dict(t=5, b=5, l=5, r=5),
    )
    return fig
