"""components/cards.py — Metric and storyline cards."""
import streamlit as st
from config import BG_SECONDARY, BORDER_COLOR, TEXT_COLOR


def metric_card(title: str, value: str, delta: str = "",
                color: str = "#4488ff", icon: str = "") -> None:
    st.markdown(
        f"""<div style="
            background:{BG_SECONDARY};border:1px solid {BORDER_COLOR};
            border-left:4px solid {color};border-radius:8px;
            padding:12px 16px;margin-bottom:8px;">
          <div style="font-size:0.75em;color:#8b949e;text-transform:uppercase;
                      letter-spacing:0.05em;">{icon} {title}</div>
          <div style="font-size:1.6em;font-weight:700;color:{color};
                      margin:4px 0;">{value}</div>
          <div style="font-size:0.8em;color:#8b949e;">{delta}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def story_card(card: dict) -> None:
    """Render one storyline card (from storyline.json cards[])."""
    icon  = card.get("icon", "")
    title = card.get("title", "")
    value = card.get("value") or ""
    text  = card.get("text", "")
    step  = card.get("step", "")

    colors = {1: "#4488ff", 2: "#2ecc71", 3: "#f39c12", 4: "#e74c3c"}
    color  = colors.get(step, "#888888")

    st.markdown(
        f"""<div style="
            background:{BG_SECONDARY};border:1px solid {BORDER_COLOR};
            border-left:4px solid {color};border-radius:10px;
            padding:16px 20px;margin-bottom:12px;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
            <span style="font-size:1.4em">{icon}</span>
            <span style="font-size:0.8em;color:#8b949e;text-transform:uppercase;
                         letter-spacing:0.08em;">Step {step} — {title}</span>
            {"<span style='margin-left:auto;font-size:1.4em;font-weight:700;color:"+color+"'>"+value+"</span>" if value else ""}
          </div>
          <div style="color:{TEXT_COLOR};font-size:0.92em;line-height:1.6;">{text}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def info_banner(text: str, color: str = "#1c3a5e") -> None:
    st.markdown(
        f"""<div style="background:{color};border-radius:8px;
            padding:12px 16px;margin-bottom:12px;
            font-size:0.9em;color:{TEXT_COLOR};">{text}</div>""",
        unsafe_allow_html=True,
    )
