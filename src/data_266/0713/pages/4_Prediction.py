"""Page 4 — Prediction: Full GNN galaxy graph + query selector + top-K predictions."""
from __future__ import annotations
import sys, os, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from pyvis.network import Network

from utils.loader import load_full_graph, load_embeddings, load_prediction, load_graph
from components.graph import build_pyvis
from components.gauge import probability_gauge
from components.cards import info_banner
from config import BG_COLOR, BG_SECONDARY, BORDER_COLOR, TEXT_COLOR

st.set_page_config(page_title="Prediction · Patent XAI", layout="wide")

# ── Color palette (12 companies + default gray) ───────────────────────────────
CO_PALETTE = [
    "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
    "#1abc9c","#e67e22","#607d8b","#e91e63","#00bcd4",
    "#8bc34a","#ff5722",
]
PLOTLY_BASE = dict(
    plot_bgcolor  = BG_COLOR,
    paper_bgcolor = BG_COLOR,
    font_color    = TEXT_COLOR,
)

def _co_color(co_idx: int) -> str:
    if 0 <= co_idx < len(CO_PALETTE):
        return CO_PALETTE[co_idx]
    return "#3a3f4a"


# ── Build full-graph PyVis (galaxy) HTML ──────────────────────────────────────
def _galaxy_html(
    all_nodes: list,
    lit_edges: list,
    src_idx: int,
    top_k_list: list,
    gt_rank_info: dict,
    height: str = "680px",
) -> str:
    query_id = f"pat_{src_idx}"
    top_ids  = {f"pat_{t['idx']}" for t in top_k_list}
    gt_ids   = {f"pat_{t['idx']}" for t in top_k_list if t["is_gt"]}
    top_probs = {f"pat_{t['idx']}": t["prob"] for t in top_k_list}
    top_rank  = {f"pat_{t['idx']}": i + 1 for i, t in enumerate(top_k_list)}

    net = Network(height=height, width="100%",
                  bgcolor="#0d1117", font_color="#e6edf3",
                  directed=False, notebook=False)
    net.set_options("""{
      "physics": {"enabled": false},
      "interaction": {"hover": true, "zoomView": true, "navigationButtons": true},
      "nodes": {"borderWidth": 1, "shadow": {"enabled": false}},
      "edges": {"smooth": {"type": "continuous"}}
    }""")

    # Add all patent nodes
    for n in all_nodes:
        nid    = n["id"]
        is_q   = nid == query_id
        is_gt  = nid in gt_ids
        is_top = nid in top_ids
        co_idx = n.get("co_idx", -1)

        if is_q:
            color, border, size, shape = "#ff4b4b", "#ffffff", 28, "star"
            label      = n["short_label"]
            font_size  = 12
        elif is_gt:
            color, border, size, shape = "#f1c40f", "#ffffff", 18, "diamond"
            label      = n["short_label"]
            font_size  = 10
        elif is_top:
            prob       = top_probs.get(nid, 0.4)
            color      = "#5d6b98"
            border     = "#c678dd"
            size       = max(7, int(prob * 14))
            shape      = "dot"
            label      = ""
            font_size  = 0
        else:
            color = _co_color(co_idx)
            border, size, shape = color, 4, "dot"
            label, font_size = "", 0

        if is_q:
            tip = f"<b>{n['label']}</b><br>{n.get('owner','')[:30]}<br>⭐ QUERY PATENT"
        elif is_gt:
            gi   = int(nid.split("_")[1])
            rk   = gt_rank_info.get(gi, {}).get("rank", "?")
            prob = gt_rank_info.get(gi, {}).get("prob", 0)
            tip  = (f"<b>{n['label']}</b><br>{n.get('owner','')[:30]}<br>"
                    f"🟡 GROUND TRUTH<br>Rank #{rk} / {len(all_nodes)}  P={prob:.4f}")
        elif is_top:
            prob = top_probs.get(nid, 0)
            rk   = top_rank.get(nid, "?")
            tip  = (f"<b>{n['label']}</b><br>{n.get('owner','')[:30]}<br>"
                    f"📌 Predicted #{rk}  P={prob:.4f}")
        else:
            tip = f"<b>{n['label']}</b><br>{n.get('owner','')[:30]}"

        net.add_node(
            nid, x=n["x"], y=n["y"], physics=False,
            label=label,
            color={"background": color, "border": border},
            shape=shape, size=size, title=tip,
            font={"size": font_size, "color": "#e6edf3"},
        )

    # Litigation background edges (thin dark red)
    for e in lit_edges:
        net.add_edge(
            f"pat_{e['s']}", f"pat_{e['d']}",
            color={"color": "#5a1a1a", "opacity": 0.35},
            width=0.7,
        )

    # Predicted edges from query
    for t in top_k_list:
        nid = f"pat_{t['idx']}"
        col = "#f1c40f" if t["is_gt"] else "#c678dd"
        wid = 3.0 if t["is_gt"] else max(0.8, t["prob"] * 5)
        dash = not t["is_gt"]
        net.add_edge(
            query_id, nid,
            color={"color": col, "opacity": 0.92 if t["is_gt"] else 0.60},
            width=wid, dashes=dash,
        )

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False,
                                     mode="w", encoding="utf-8") as tmp:
        net.save_graph(tmp.name)
        html = open(tmp.name, encoding="utf-8").read()
    os.unlink(tmp.name)
    return html


# ── Load data ─────────────────────────────────────────────────────────────────
full  = load_full_graph()
h_emb = load_embeddings()
pred  = load_prediction()
graph = load_graph()

if not full or h_emb is None:
    st.error("Run: `conda run -n durendal python compute/run_all.py --pair 0 --khop 2 --topk 20`")
    st.stop()

all_nodes       = full.get("nodes", [])
lit_edges       = full.get("litigation_edges", [])
test_plaintiffs = full.get("test_plaintiffs", [])
top_cos         = full.get("metadata", {}).get("top_companies", [])
N               = len(all_nodes)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Prediction")
    top_k_n = st.slider("Top-K shown", 5, 30, 20)
    st.markdown("---")
    st.markdown(f"**Test plaintiffs**: {len(test_plaintiffs)}")
    st.markdown(f"**Total patents**: {N}")
    if pred:
        rank = pred.get("rank", "?")
        total = pred.get("total_patents", N)
        st.markdown("**Pre-computed pair 0**")
        st.markdown(f"`{pred.get('query_patent','?')}` → `{pred.get('target_patent','?')}`")
        st.metric("GT Rank", f"#{rank} / {total}")

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("# 🎯 Prediction")

tab1, tab2 = st.tabs(["🌐 Full GNN Graph", "🔬 XAI Context (k-hop)"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1  —  Full GNN Galaxy with query selector
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Query selector ────────────────────────────────────────────────────────
    options = [
        f"#{i}  {tp['label']}  ({tp.get('owner','?')[:22]})"
        for i, tp in enumerate(test_plaintiffs)
    ]
    sel_idx = st.selectbox(
        "Query Patent (select from test plaintiff list)",
        range(len(options)),
        format_func=lambda i: options[i],
        index=0,
    )

    query   = test_plaintiffs[sel_idx]
    src_idx = query["src_idx"]
    gt_list = query.get("gts", [])
    gt_idxs = {g["dst_idx"] for g in gt_list}

    # Header card
    gt_labels = "  |  ".join(f"🟡 {g['gt_label']}" for g in gt_list[:3])
    st.markdown(
        f"""<div style="background:{BG_SECONDARY};border:1px solid {BORDER_COLOR};
            border-left:4px solid #ff4b4b;border-radius:8px;
            padding:10px 18px;margin-bottom:12px;">
          <b style="color:#ff4b4b;font-size:1.1em">⭐ {query['label']}</b>
          <span style="color:#8b949e"> — {query.get('owner','?')}</span>
          &nbsp;&nbsp;→&nbsp;&nbsp;{gt_labels}
        </div>""",
        unsafe_allow_html=True,
    )

    # ── On-the-fly top-K prediction ───────────────────────────────────────────
    logits = h_emb @ h_emb[src_idx]
    logits[src_idx] = -np.inf
    probs  = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))

    top_indices = np.argsort(-probs)
    top_k_list  = []
    for idx in top_indices:
        i = int(idx)
        if i == src_idx:
            continue
        top_k_list.append({
            "idx":   i,
            "prob":  float(probs[i]),
            "is_gt": i in gt_idxs,
            "label": all_nodes[i]["label"] if i < N else f"pat_{i}",
            "owner": all_nodes[i].get("owner", "") if i < N else "",
        })
        if len(top_k_list) >= top_k_n:
            break

    # Force-include GT defendants that are outside top-K
    missing = [g for g in gt_list if g["dst_idx"] not in {t["idx"] for t in top_k_list}]
    for g in missing:
        i = g["dst_idx"]
        top_k_list.append({
            "idx": i, "prob": float(probs[i]), "is_gt": True,
            "label": all_nodes[i]["label"] if i < N else f"pat_{i}",
            "owner": all_nodes[i].get("owner", "") if i < N else "",
        })

    # GT rank info (actual rank in full 1331 ranking)
    gt_rank_info: dict = {}
    for g in gt_list:
        gi = g["dst_idx"]
        gt_rank_info[gi] = {
            "rank": int((probs > probs[gi]).sum()) + 1,
            "prob": float(probs[gi]),
        }

    # ── Layout: galaxy graph (left) + ranking (right) ─────────────────────────
    col_g, col_r = st.columns([3, 2])

    with col_g:
        st.markdown("#### GNN Embedding Space — Patent Galaxy")
        st.caption(
            f"Spring layout of all {N} patents. "
            "Dark red lines = existing lawsuits. "
            "Select a plaintiff → see predicted lawsuit edges."
        )

        galaxy_html = _galaxy_html(
            all_nodes, lit_edges, src_idx, top_k_list, gt_rank_info,
            height="660px",
        )
        components.html(galaxy_html, height=680)

        # Legend row
        co_legend = "".join(
            f'<span style="display:inline-flex;align-items:center;margin-right:10px;">'
            f'<span style="width:9px;height:9px;border-radius:2px;background:{CO_PALETTE[i]};'
            f'display:inline-block;margin-right:4px;"></span>'
            f'<span style="font-size:0.72em;color:#8b949e">{co[:15]}</span></span>'
            for i, co in enumerate(top_cos[:8])
        )
        st.markdown(
            f'<div style="margin-top:4px">{co_legend}'
            f'&nbsp;&nbsp;⭐ Query &nbsp;&nbsp;🟡 Ground Truth &nbsp;&nbsp;● Predicted</div>',
            unsafe_allow_html=True,
        )

    with col_r:
        st.markdown("#### Top-K Ranking")

        # GT rank summary cards
        for gi, info in gt_rank_info.items():
            pc = "#2ecc71" if info["rank"] <= 20 else "#f39c12" if info["rank"] <= 100 else "#e74c3c"
            gl = all_nodes[gi]["label"] if gi < N else f"pat_{gi}"
            go_ = all_nodes[gi].get("owner", "—") if gi < N else "—"
            st.markdown(
                f"""<div style="background:{BG_SECONDARY};border:1px solid {BORDER_COLOR};
                    border-left:4px solid {pc};border-radius:8px;
                    padding:8px 14px;margin-bottom:8px">
                  <div style="color:#8b949e;font-size:0.72em">GROUND TRUTH RANK</div>
                  <b style="color:#f1c40f">{gl}</b>
                  <span style="color:#8b949e;font-size:0.8em"> — {go_[:22]}</span><br>
                  <span style="font-size:1.8em;font-weight:800;color:{pc}">#{info['rank']}</span>
                  <span style="color:#8b949e;font-size:0.85em"> / {N}</span>
                  &nbsp;&nbsp;<span style="color:#8b949e;font-size:0.8em">P = {info['prob']:.4f}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        # Bar chart
        display = top_k_list[:top_k_n]
        if display:
            bar_labels = []
            for i, t in enumerate(display):
                gt_m = " ←GT" if t["is_gt"] else ""
                bar_labels.append(f"#{i+1} {t['label'][:11]}{gt_m}")
            vals   = [t["prob"] for t in display]
            colors = ["#f1c40f" if t["is_gt"] else "#5d6b98" for t in display]

            fig = go.Figure(go.Bar(
                x=vals, y=bar_labels, orientation="h",
                marker_color=colors,
                text=[f"{v:.3f}" for v in vals], textposition="outside",
            ))
            fig.update_layout(
                **PLOTLY_BASE,
                height=max(280, 26 * len(display) + 50),
                margin=dict(t=36, b=16, l=150, r=50),
                xaxis=dict(range=[0, max(vals) * 1.18 if vals else 1],
                           title="P(sue)", color=TEXT_COLOR),
                yaxis=dict(autorange="reversed", tickfont=dict(size=9), color=TEXT_COLOR),
                title=dict(text=f"Top-{top_k_n}   🟡 = Ground Truth", font=dict(size=12)),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Table
        import pandas as pd
        rows = [
            {
                "Rank":   f"#{i+1}",
                "Patent": t["label"],
                "Owner":  t["owner"][:20] if t["owner"] else "—",
                "P(sue)": f"{t['prob']:.4f}",
                "GT":     "🟡 ✓" if t["is_gt"] else "",
            }
            for i, t in enumerate(display)
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    info_banner(
        "💡 The model scores all 1,331 patents as potential defendants for the selected query patent. "
        "🟡 Gold = actual ground truth (real lawsuit). → <b>🔍 Explanation</b> tab shows XAI on the ground truth pair.",
        color="#1c2a3a",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2  —  XAI context: k-hop subgraph with XAI overlays (pre-computed pair 0)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not pred:
        st.info("Run pipeline first."); st.stop()

    nodes_kh = graph.get("nodes", []) if graph else []
    edges_kh = graph.get("edges", []) if graph else []
    p     = pred.get("probability", 0.0)
    thr   = pred.get("threshold",   0.5)
    rank  = pred.get("rank", "?")
    total = pred.get("total_patents", N)
    pct   = pred.get("rank_percentile", 0.0)
    topk  = pred.get("top_k", [])

    st.markdown("#### Pre-computed Pair (pair=0) — XAI Context Graph")
    st.caption(
        f"k-hop subgraph around **{pred.get('query_patent','?')}** → "
        f"**{pred.get('target_patent','?')}**. "
        f"Ground truth ranked #{rank}/{total} (P={p:.4f}). XAI overlays below."
    )

    col_cg, col_cr = st.columns([3, 2])

    with col_cg:
        mode_opt = st.radio(
            "Graph mode",
            ["Prediction overlay", "GNN importance", "Attention"],
            horizontal=True, key="tab2_mode",
        )
        show_co  = st.checkbox("Company nodes", value=False, key="t2co")
        show_ipc = st.checkbox("IPC nodes",     value=False, key="t2ipc")

        mode_map = {
            "Prediction overlay": ("importance", True,  True),
            "GNN importance":     ("importance", False, False),
            "Attention":          ("attn_layer2",False, False),
        }
        graph_mode, show_pred, pred_mode = mode_map[mode_opt]

        if nodes_kh:
            html_kh = build_pyvis(
                nodes_kh, edges_kh, height="520px",
                mode=graph_mode, show_company=show_co, show_ipc=show_ipc,
                show_predicted=show_pred, prediction_mode=pred_mode, node_scale=26,
            )
            components.html(html_kh, height=540)
        else:
            st.warning("k-hop subgraph not found. Run pipeline first.")

    with col_cr:
        correct = pred.get("correct", False)
        vc = "#2ecc71" if correct else "#e74c3c"
        pc2 = "#2ecc71" if isinstance(rank, int) and rank <= 20 else "#f39c12"
        st.markdown(
            f"""<div style="background:{BG_SECONDARY};border:1px solid {BORDER_COLOR};
                border-left:4px solid #ff4b4b;border-radius:8px;
                padding:12px 16px;margin-bottom:10px">
              <div style="color:#8b949e;font-size:0.72em;text-transform:uppercase">INPUT PAIR</div>
              <div style="color:#ff4b4b;font-weight:700">⭐ {pred.get('query_patent','?')}</div>
              <div style="color:#8b949e;font-size:0.8em">{pred.get('plaintiff_company','—')}</div>
              <div style="color:#8b949e;margin:3px 0">→</div>
              <div style="color:#f1c40f;font-weight:700">🟡 {pred.get('target_patent','?')}</div>
              <div style="color:#8b949e;font-size:0.8em">{pred.get('defendant_company','—')}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<div style="background:{BG_SECONDARY};border:1px solid {BORDER_COLOR};
                border-left:4px solid {vc};border-radius:8px;
                padding:12px 16px;margin-bottom:10px">
              <div style="color:#8b949e;font-size:0.72em;text-transform:uppercase">EVALUATION</div>
              <div style="color:{vc};font-weight:700">
                {'✓ Correct' if correct else '✗ Incorrect'}</div>
              <div style="margin:6px 0">
                <span style="font-size:1.8em;font-weight:800;color:{pc2}">#{rank}</span>
                <span style="color:#8b949e"> / {total}</span>
              </div>
              <div style="color:#8b949e;font-size:0.78em">Top {100-pct:.1f}% &nbsp;|&nbsp; P={p:.4f} &nbsp;|&nbsp; thr={thr:.4f}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.plotly_chart(probability_gauge(p, thr, height=200), use_container_width=True)

        if topk:
            display_t = topk[:10]
            if not any(t["is_ground_truth"] for t in display_t):
                gt_e = next((t for t in topk if t["is_ground_truth"]), None)
                if gt_e:
                    display_t = display_t + [gt_e]
            lb = [f"#{t['rank']} {t['patent_label'][:10]}" + (" ←GT" if t["is_ground_truth"] else "")
                  for t in display_t]
            vt = [float(t["probability"]) for t in display_t]
            ct = ["#f1c40f" if t["is_ground_truth"] else "#5d6b98" for t in display_t]
            fig_t = go.Figure(go.Bar(
                x=vt, y=lb, orientation="h", marker_color=ct,
                text=[f"{v:.3f}" for v in vt], textposition="outside",
            ))
            fig_t.update_layout(
                **PLOTLY_BASE,
                height=max(200, 26 * len(display_t) + 40),
                margin=dict(t=32, b=16, l=125, r=40),
                xaxis=dict(range=[0, 1.15], title="P(sue)", color=TEXT_COLOR),
                yaxis=dict(autorange="reversed", tickfont=dict(size=8), color=TEXT_COLOR),
                title=dict(text="Top-10 (pair 0)", font=dict(size=11)),
            )
            st.plotly_chart(fig_t, use_container_width=True)
