"""app.py — Patent Litigation XAI Dashboard · Home page & sidebar."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import numpy as np
from utils.loader import load_prediction, load_metadata, check_results
from components.gauge import probability_gauge, confidence_bar
from components.legend import render_full_legend
from config import BG_COLOR, TEXT_COLOR, BG_SECONDARY

st.set_page_config(
    page_title="Patent XAI Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(f"""<style>
  .block-container {{padding-top:0.8rem;}}
  section[data-testid="stSidebar"] {{background:{BG_SECONDARY};}}
  div[data-testid="metric-container"] {{
    border:1px solid #30363d;border-radius:8px;padding:8px;
  }}
</style>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Patent XAI")
    st.markdown("---")

    missing = check_results()
    if missing:
        st.error(f"Missing: {', '.join(missing)}")
        st.code("python compute/run_all.py --pair 0")
        st.stop()

    pred = load_prediction()
    meta = load_metadata()

    p    = pred.get("probability", 0.0)
    pc   = "#e74c3c" if p > 0.7 else "#f39c12" if p > 0.5 else "#2ecc71"

    st.markdown(
        f"**Plaintiff**\n\n`{pred.get('query_patent','?')}`"
        f"\n\n{pred.get('plaintiff_company','—')}",
    )
    st.markdown(
        f"**Defendant**\n\n`{pred.get('target_patent','?')}`"
        f"\n\n{pred.get('defendant_company','—')}",
    )
    st.markdown(
        f"**P(sue) = <span style='color:{pc};font-size:1.4em;font-weight:700'>"
        f"{p:.4f}</span>**",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"Prediction: **{'⚖️ SUE' if pred.get('prediction')=='sue' else '✅ NO SUE'}**  "
        f"· Confidence: **{pred.get('confidence','?').upper()}**"
    )
    st.markdown("---")

    st.markdown("### Navigation")
    st.page_link("pages/1_Overview.py",        label="1. Overview",        icon="🌐")
    st.page_link("pages/2_Integration.py",     label="2. Integration",     icon="🔗")
    st.page_link("pages/3_Message_Passing.py", label="3. Message Passing", icon="📡")
    st.page_link("pages/4_Prediction.py",      label="4. Prediction",      icon="🎯")
    st.page_link("pages/5_Explanation.py",     label="5. Explanation",     icon="🔍")
    st.page_link("pages/6_Counterfactual.py",  label="6. Counterfactual",  icon="🔄")
    st.page_link("pages/7_Conclusion.py",      label="7. Conclusion",      icon="✅")

    st.markdown("---")
    st.markdown("### Legend")
    render_full_legend()

    st.markdown("---")
    if meta:
        st.caption(
            f"Model: {meta.get('model','?')}\n"
            f"Pair: {meta.get('pair','?')} · K={meta.get('khop','?')}"
        )

# ── Home page content ──────────────────────────────────────────────────────────
st.markdown("# ⚖️ Patent Litigation XAI Dashboard")
st.markdown(
    "This dashboard explains **why the GNN predicted a litigation link** "
    "between two patents, step by step."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("P(sue)", f"{p:.4f}", f"{'▲ above' if p >= pred.get('threshold',0.5) else '▼ below'} {pred.get('threshold',0.5):.2f}")
col2.metric("Prediction",  pred.get("prediction","?").upper())
col3.metric("Confidence",  pred.get("confidence","?").upper())
col4.metric("Target Rank", f"#{pred.get('target_rank','?')}")

st.markdown("---")

col_gauge, col_flow = st.columns([1, 2])

with col_gauge:
    st.plotly_chart(probability_gauge(p, pred.get("threshold", 0.5)), use_container_width=True)
    st.plotly_chart(confidence_bar(p), use_container_width=True)

with col_flow:
    st.markdown("### Story")
    st.markdown("""
| Step | Page | Question |
|------|------|---------|
| 1 | 🌐 Overview | What did the GNN learn? |
| 2 | 🔗 Integration | How is a new patent inserted? |
| 3 | 📡 Message Passing | How does information flow? |
| 4 | 🎯 Prediction | Who is predicted to be sued? |
| 5 | 🔍 Explanation | Why was this predicted? |
| 6 | 🔄 Counterfactual | What changes the result? |
| 7 | ✅ Conclusion | Final explanation |
""")
    st.info("Use the **sidebar** or the page links above to navigate.")

if meta:
    st.markdown("---")
    st.markdown("### Model Info")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model",   meta.get("model","?"))
    c2.metric("Patents", f"{meta.get('n_patents','?'):,}")
    c3.metric("Layers",  meta.get("layers","?"))
    c4.metric("Heads",   str(meta.get("heads","?")))
