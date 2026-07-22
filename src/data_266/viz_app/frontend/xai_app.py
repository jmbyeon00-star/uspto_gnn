"""xai_app.py — GNN XAI 해석 Streamlit 앱 (port 8502)
7가지 XAI 방법으로 특허 소송 링크 예측을 해석합니다.
"""
import sys, warnings, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import defaultdict
from scipy.special import comb as sp_comb
import streamlit as st

if not hasattr(np, '_core'):
    import numpy.core as _nc
    sys.modules['numpy._core'] = _nc
    sys.modules['numpy._core.multiarray'] = _nc.multiarray

from backend.model import SimpleCatHeteroGAT, EDGE_TYPES

# ── 페이지 설정 ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patent GNN XAI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
  html, body, [class*="css"] {
    background-color: #050810 !important;
    color: #e8eaf0;
    font-family: 'JetBrains Mono', monospace;
  }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070B18, #0A0E1A);
    border-right: 1px solid rgba(79,195,247,0.15);
  }
  /* ── glass card ── */
  .glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(14px);
    margin-bottom: 14px;
    transition: border-color .25s;
  }
  .glass-card:hover { border-color: rgba(79,195,247,0.35); }
  /* ── XAI section header ── */
  .method-header {
    font-size: 1.05rem; font-weight: 700;
    color: #4fc3f7; margin-bottom: 0.4rem;
    letter-spacing: .06em; text-transform: uppercase;
  }
  /* ── info box ── */
  .desc-box {
    background: rgba(79,195,247,0.05);
    border-left: 3px solid #4fc3f7;
    padding: 0.7rem 1rem; border-radius: 6px;
    margin-bottom: 1rem; font-size: 0.88rem;
    color: #b0bec5;
  }
  /* ── metric cards ── */
  .metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 1rem;
    text-align: center; margin: 0.3rem;
  }
  .metric-label {
    font-size: .7rem; color: #546e7a;
    letter-spacing: .12em; text-transform: uppercase; margin-bottom: 4px;
  }
  .metric-cyan {
    font-size: 1.9rem; font-weight: 700;
    background: linear-gradient(135deg,#4fc3f7,#00BFFF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .metric-gold { font-size: 1.9rem; font-weight: 700; color: #FFD700; }
  /* ── prediction badge ── */
  .pred-high { color: #ef5350; font-size: 1.5rem; font-weight: 700; }
  .pred-low  { color: #42a5f5; font-size: 1.5rem; font-weight: 700; }
  /* ── buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, rgba(79,195,247,0.1), rgba(0,191,255,0.1));
    border: 1px solid rgba(79,195,247,0.4);
    color: #4fc3f7; border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700; transition: all .2s;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, rgba(79,195,247,0.22), rgba(0,191,255,0.22));
    border-color: #4fc3f7; box-shadow: 0 0 14px rgba(79,195,247,0.3);
  }
  /* ── slider ── */
  [data-testid="stSlider"] > div > div > div { background: #4fc3f766; }
  /* ── tab bar ── */
  [data-testid="stTabs"] [role="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: .82rem; font-weight: 700;
  }
  [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #4fc3f7; border-bottom: 2px solid #4fc3f7;
  }
  /* ── divider ── */
  hr { border-color: rgba(255,255,255,0.05) !important; }
  /* ── scrollbar ── */
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: #050810; }
  ::-webkit-scrollbar-thumb { background: rgba(79,195,247,0.25); border-radius: 3px; }
  /* ── dataframe ── */
  [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
  /* ── expander ── */
  [data-testid="stExpander"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
  }
</style>
""", unsafe_allow_html=True)


# ── 모델 로드 (캐시) ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="모델 로딩 중...")
def load_model():
    import pathlib
    base = pathlib.Path(__file__).parent.parent.parent
    ckpt = torch.load(base / 'phase3_simplecat_ckpt.pt', map_location='cpu', weights_only=False)
    data = torch.load(base / 'hetero_data.pt',            map_location='cpu', weights_only=False)
    model = SimpleCatHeteroGAT(n_ipc=len(ckpt['ipc_list']))
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    with torch.no_grad():
        h_pat = model(data)['patent'].detach().cpu()
    return model, data, h_pat, ckpt


model, data, h_pat, ckpt = load_model()
pat_list  = ckpt['pat_list']
ipc_list  = ckpt['ipc_list']
co_list   = ckpt['co_list']
best_thr  = float(ckpt['best_threshold'])
sued_e    = np.array(ckpt['sued_edges'])
te_idx    = np.array(ckpt['test_edge_idx'])
test_edges = sued_e[:, te_idx]
h_np      = h_pat.numpy()


# ── test 쌍 예측 확률 계산 (캐시) ──────────────────────────────────────────
@st.cache_data
def get_test_pairs():
    pairs = []
    for j in range(test_edges.shape[1]):
        s, d = int(test_edges[0, j]), int(test_edges[1, j])
        logit = float(np.dot(h_np[s], h_np[d]))
        prob  = 1.0 / (1.0 + np.exp(-logit))
        pairs.append((s, d, prob))
    pairs.sort(key=lambda x: -x[2])
    return pairs

test_pairs = get_test_pairs()


# ── 공통 유틸 ───────────────────────────────────────────────────────────────
def node_label(idx, ntype):
    if ntype == 'patent': return str(pat_list[idx])
    if ntype == 'ipc':    return ipc_list[idx] if idx < len(ipc_list) else f'IPC{idx}'
    return co_list[idx] if idx < len(co_list) else f'CO{idx}'

def get_patent_neighbors(node_idx):
    nb = set()
    for et in EDGE_TYPES:
        src_t, _, dst_t = et
        if et not in data.edge_index_dict: continue
        eidx = data.edge_index_dict[et].numpy()
        if src_t == 'patent' and dst_t == 'patent':
            nb.update(eidx[1][eidx[0] == node_idx].tolist())
            nb.update(eidx[0][eidx[1] == node_idx].tolist())
    return nb - {node_idx}


def compute_attention(target_idx):
    x_pat_raw = torch.cat([data['patent'].struct, data['patent'].x], dim=1)
    x = {
        'patent':  model.pat_proj(x_pat_raw),
        'company': F.leaky_relu(model.co_proj(data['company'].x)),
        'ipc':     model.ipc_emb.weight.clone(),
    }
    acts = [F.leaky_relu, F.leaky_relu, torch.tanh]
    all_attn = {}
    for li, (hconv, bn, act) in enumerate(zip(model.convs[:3], model.bns, acts)):
        x_agg = defaultdict(list)
        layer_attn = {}
        for et in EDGE_TYPES:
            src_t, rel, dst_t = et
            if et not in data.edge_index_dict or et not in hconv.convs: continue
            et_conv  = hconv.convs[et]
            edge_idx = data.edge_index_dict[et]
            with torch.no_grad():
                out, (ei_ret, alpha) = et_conv(
                    (x[src_t], x[dst_t]), edge_idx, return_attention_weights=True)
            if dst_t == 'patent':
                mask = (ei_ret[1] == target_idx)
                if mask.any():
                    layer_attn[et] = {
                        'src': ei_ret[0][mask].cpu().tolist(),
                        'attn': alpha[mask].mean(dim=1).cpu().tolist()
                    }
            x_agg[dst_t].append(out)
        all_attn[f'L{li+1}'] = layer_attn
        x_new = {k: torch.stack(vs).mean(0) for k, vs in x_agg.items()}
        x = {k: model.drop(act(bn[k](v.view(-1, model.hid)))) if k in bn
             else model.drop(act(v)) for k, v in x_new.items()}
    return all_attn


def compute_attention_flow(target_idx: int, top_per_hop: int = 20) -> dict:
    """전체 attention matrix를 레이어별로 저장 후 역추적.
    반환: {1: [(node, flow),...], 2:[...], 3:[...]}
    hop 1 = L3 backward (직접 1-hop), hop 3 = L1 backward (원거리 3-hop)
    """
    x_pat_raw = torch.cat([data['patent'].struct, data['patent'].x], dim=1)
    x = {
        'patent':  model.pat_proj(x_pat_raw),
        'company': F.leaky_relu(model.co_proj(data['company'].x)),
        'ipc':     model.ipc_emb.weight.clone(),
    }
    acts = [F.leaky_relu, F.leaky_relu, torch.tanh]
    dst_to_srcs = []  # layer_i → {dst: [(src, attn)]}

    for li, (hconv, bn, act) in enumerate(zip(model.convs[:3], model.bns, acts)):
        x_agg = defaultdict(list)
        d2s   = defaultdict(list)
        for et in EDGE_TYPES:
            src_t, rel, dst_t = et
            if et not in data.edge_index_dict or et not in hconv.convs: continue
            et_conv  = hconv.convs[et]
            edge_idx = data.edge_index_dict[et]
            with torch.no_grad():
                out, (ei_ret, alpha) = et_conv(
                    (x[src_t], x[dst_t]), edge_idx, return_attention_weights=True)
            if src_t == 'patent' and dst_t == 'patent':
                am   = alpha.mean(dim=1).cpu().numpy()
                srcs = ei_ret[0].cpu().numpy()
                dsts = ei_ret[1].cpu().numpy()
                for s, d, a in zip(srcs, dsts, am):
                    d2s[int(d)].append((int(s), float(a)))
            x_agg[dst_t].append(out)
        dst_to_srcs.append(dict(d2s))
        x_new = {k: torch.stack(vs).mean(0) for k, vs in x_agg.items() if vs}
        x = {k: (model.drop(act(bn[k](v.view(-1, model.hid)))) if k in bn
                 else model.drop(act(v))) for k, v in x_new.items()}

    # 역추적: layer 2→1→0, hop 1→2→3
    current = {target_idx: 1.0}
    results = {}
    for li in range(2, -1, -1):
        new_flow: dict = {}
        for dst_n, dst_w in current.items():
            for src_n, a_w in dst_to_srcs[li].get(dst_n, []):
                if src_n == target_idx: continue
                new_flow[src_n] = new_flow.get(src_n, 0.0) + dst_w * a_w
        hop = 3 - li  # li=2→hop1, li=1→hop2, li=0→hop3
        results[hop] = sorted(new_flow.items(), key=lambda x: -x[1])[:top_per_hop]
        current = new_flow
    return results


# ── 사이드바 ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px;'>
      <div style='font-size:1.6rem; font-weight:700;
                  background:linear-gradient(135deg,#4fc3f7,#00BFFF);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                  letter-spacing:.04em;'>🔬 GNN XAI</div>
      <div style='font-size:.7rem; color:#546e7a; margin-top:4px;
                  letter-spacing:.1em;'>SimpleCatHeteroGAT</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    pair_options = [
        f"#{i+1}  {str(pat_list[s])[:16]} → {str(pat_list[d])[:16]}  ({p:.3f})"
        for i, (s, d, p) in enumerate(test_pairs)
    ]
    sel_idx = st.selectbox("📋 분석 대상 쌍", range(len(pair_options)),
                           format_func=lambda x: pair_options[x])
    QUERY_S, QUERY_D, QUERY_P = test_pairs[sel_idx]

    is_pred = QUERY_P >= best_thr
    pred_color = "#ef5350" if is_pred else "#42a5f5"
    pred_icon  = "🔴 소송 예측" if is_pred else "🔵 미예측"
    st.markdown(f"""
    <div class='glass-card' style='padding:14px 16px; margin-top:8px;'>
      <div style='font-size:.68rem; color:#546e7a; letter-spacing:.1em; text-transform:uppercase; margin-bottom:8px;'>선택된 쌍</div>
      <div style='font-size:.8rem; color:#b0bec5; margin-bottom:4px;'>
        <span style='color:#4fc3f7;'>★ PLT</span>&nbsp; {pat_list[QUERY_S]}
      </div>
      <div style='font-size:.8rem; color:#b0bec5; margin-bottom:10px;'>
        <span style='color:#ff8800;'>◆ DEF</span>&nbsp; {pat_list[QUERY_D]}
      </div>
      <div style='display:flex; justify-content:space-between; align-items:center;'>
        <div>
          <div style='font-size:.65rem; color:#546e7a;'>P(sue)</div>
          <div style='font-size:1.5rem; font-weight:700; color:{pred_color};'>{QUERY_P:.3f}</div>
        </div>
        <div style='text-align:right;'>
          <div style='font-size:.65rem; color:#546e7a;'>판정</div>
          <div style='font-size:.82rem; font-weight:700; color:{pred_color};'>{pred_icon}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("<div style='font-size:.72rem; color:#546e7a; letter-spacing:.1em; text-transform:uppercase; margin-bottom:8px;'>⚙️ 분석 방법 선택</div>", unsafe_allow_html=True)
    run_m1 = st.checkbox("M1 Attention",              value=True)
    run_m2 = st.checkbox("M2 Grad-CAM",               value=True)
    run_m3 = st.checkbox("M3 Counterfactual",          value=True)
    run_m4 = st.checkbox("M4 GNNExplainer  (~30초)",  value=False)
    run_m5 = st.checkbox("M5 PGExplainer   (~20초)",  value=False)
    run_m6 = st.checkbox("M6 SubgraphX     (~2분)",   value=False)
    run_m7 = st.checkbox("M7 GraphSHAP     (~1분)",   value=False)
    run_btn = st.button("🚀 분석 실행", type="primary", use_container_width=True)

    st.divider()
    above = sum(1 for _, _, p in test_pairs if p >= best_thr)
    st.markdown(f"""
    <div style='font-size:.72rem; color:#546e7a;'>
      Test 엣지 총 <b style='color:#b0bec5;'>{len(test_pairs)}</b>개 &nbsp;|&nbsp;
      예측 <b style='color:#ef5350;'>{above}</b>개
    </div>
    """, unsafe_allow_html=True)


# ── 메인 타이틀 ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:24px 0 8px;'>
  <div style='font-size:1.7rem; font-weight:700; letter-spacing:.04em;
              background:linear-gradient(135deg,#4fc3f7,#00BFFF,#4fc3f7);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    🔬 특허 소송 링크 예측 — XAI 해석
  </div>
  <div style='font-size:.78rem; color:#546e7a; margin-top:4px;'>
    SimpleCatHeteroGAT · Attention Flow · Grad-CAM · Counterfactual · GNNExplainer · SubgraphX · GraphSHAP
  </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class='metric-card'>
      <div class='metric-label'>Model</div>
      <div style='font-size:.85rem; font-weight:700; color:#4fc3f7;'>SimpleCatHeteroGAT</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class='metric-card'>
      <div class='metric-label'>Test AUROC</div>
      <div class='metric-cyan'>{ckpt['test_result']['auroc']:.4f}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class='metric-card'>
      <div class='metric-label'>Best Threshold</div>
      <div class='metric-gold'>{best_thr:.2f}</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class='metric-card'>
      <div class='metric-label'>Best Val F1</div>
      <div class='metric-cyan'>{ckpt.get('best_val_f1', 0):.4f}</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── 세션 캐시 키 (항상 초기화) ─────────────────────────────────────────────
cache_key = f"{QUERY_S}_{QUERY_D}"
if 'xai_cache' not in st.session_state:
    st.session_state.xai_cache = {}
cache = st.session_state.xai_cache.setdefault(cache_key, {})


# ══════════════════════════════════════════════════════════════════════════════
# VIZ A: Embedding Space (t-SNE + Plotly)  — 항상 표시
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🌌 Patent Embedding Space (t-SNE)")
st.markdown("GNN이 학습한 64d 임베딩을 2D로 투영 — 가까울수록 소송 관계가 예측됨")

@st.cache_data(show_spinner="t-SNE 계산 중 (최초 1회)...")
def compute_tsne(h_np_bytes, perplexity=30):
    import numpy as _np
    from sklearn.manifold import TSNE
    h = _np.frombuffer(h_np_bytes, dtype=_np.float32).reshape(-1, 64)
    return TSNE(n_components=2, perplexity=perplexity, random_state=42,
                n_iter=1000).fit_transform(h)

xy = compute_tsne(h_pat.numpy().astype('float32').tobytes())

import plotly.graph_objects as go

# 선택된 원고 기준 모든 특허의 소송 확률
all_probs = torch.sigmoid(h_pat @ h_pat[QUERY_S]).numpy()

# 소송 빈도(degree) 계산 — 많이 엮일수록 점 크게
from collections import Counter
sue_degree = Counter(sued_e[0].tolist()) + Counter(sued_e[1].tolist())
max_deg = max(sue_degree.values()) if sue_degree else 1

others = [i for i in range(len(pat_list)) if i not in (QUERY_S, QUERY_D)]
pt_sizes = [4 + 14 * (sue_degree.get(i, 0) / max_deg) for i in others]

_colorbar = dict(
    title=dict(text="P(lawsuit)", font=dict(color='white', size=11)),
    tickfont=dict(color='white'), thickness=14, len=0.7,
    bgcolor='rgba(0,0,0,0)',
)

fig_tsne = go.Figure()

# ── 일반 특허들 (색=확률, 크기=소송빈도)
fig_tsne.add_trace(go.Scatter(
    x=xy[others, 0], y=xy[others, 1], mode='markers',
    marker=dict(size=pt_sizes, color=all_probs[others], colorscale='plasma',
                cmin=0, cmax=1, opacity=0.85, line=dict(width=0),
                colorbar=_colorbar),
    text=[f"Patent: {pat_list[i]}<br>P(sue): {all_probs[i]:.4f}<br>소송 횟수: {sue_degree.get(i,0)}회"
          for i in others],
    hovertemplate="%{text}<extra></extra>",
    name="특허 (크기=소송 빈도)", showlegend=True
))

# ── 선택된 원고 (흰 별)
fig_tsne.add_trace(go.Scatter(
    x=[xy[QUERY_S, 0]], y=[xy[QUERY_S, 1]], mode='markers+text',
    marker=dict(size=26, color='white', symbol='star',
                line=dict(width=2, color='#ffdd00')),
    text=[f"PLT: {pat_list[QUERY_S]}"],
    textposition="top center", textfont=dict(color='white', size=11),
    name="★ 선택 원고 (Patent A)",
    hovertemplate=f"<b>Patent A (Plaintiff in GT)</b><br>{pat_list[QUERY_S]}<extra></extra>"
))

# ── 선택된 피고 (주황 다이아)
fig_tsne.add_trace(go.Scatter(
    x=[xy[QUERY_D, 0]], y=[xy[QUERY_D, 1]], mode='markers+text',
    marker=dict(size=26, color='#ff8800', symbol='diamond',
                line=dict(width=2, color='white')),
    text=[f"DEF: {pat_list[QUERY_D]}"],
    textposition="top center", textfont=dict(color='white', size=11),
    name="◆ 선택 피고 (Patent B)",
    hovertemplate=f"<b>Patent B (Defendant in GT)</b><br>{pat_list[QUERY_D]}<br>P(sue)={QUERY_P:.4f}<extra></extra>"
))

fig_tsne.update_layout(
    template='plotly_dark', paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
    height=620,
    title=dict(
        text=f"색 = {pat_list[QUERY_S]}의 소송 예측 확률  |  흰 테두리 = 실제 소송 이력 있음  |  ★=원고  ◆=피고",
        font=dict(color='white', size=11)
    ),
    legend=dict(bgcolor='rgba(30,33,48,0.8)', bordercolor='#444',
                font=dict(color='white', size=10)),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
)
st.plotly_chart(fig_tsne, use_container_width=True)
with st.expander("읽는 법"):
    st.markdown("""
- **색깔** = 해당 특허가 선택된 원고(★)로부터 소송당할 예측 확률 — 밝은 노랑 = 높음, 어두운 보라 = 낮음
- **색깔** = 선택된 원고(★)로부터 소송당할 예측 확률 — 밝은 노랑=높음, 어두운 보라=낮음
- **점 크기** = 실제 소송에 엮인 횟수 — 클수록 자주 소송 당사자로 등장한 특허
- **★ 흰 별** = 선택된 원고 특허 (Patent A)
- **◆ 주황 다이아** = 선택된 피고 특허 (Patent B, 실제 소송 기록)
- 피고가 밝은 색 영역에 있을수록 모델이 확신하는 예측
- **모델은 소송 방향(누가 원고/피고)을 예측하지 않음** — 쌍의 관계만 예측
- hover하면 각 특허의 P(sue) 및 소송 횟수 확인 가능
""")


# ══════════════════════════════════════════════════════════════════════════════
# VIZ B: Explanation Subgraph (PyVis)  — 항상 표시
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("## 🕸️ Explanation Subgraph (PyVis)")
st.markdown("원고-피고 주변 서브그래프 — XAI 결과(M3·M6)를 먼저 실행하면 엣지·노드에 반영됨")

import streamlit.components.v1 as components
from pyvis.network import Network

@st.cache_data
def build_explanation_graph(src_idx, dst_idx, cache_key_str,
                             cf_res_json=None, shap_json=None):
    net = Network(height="600px", width="100%", bgcolor="#0e1117",
                  font_color="white", directed=False)
    net.set_options("""{"physics":{"solver":"forceAtlas2Based",
      "forceAtlas2Based":{"gravitationalConstant":-80,"springLength":120},
      "stabilization":{"iterations":120}},
      "edges":{"smooth":{"type":"continuous"}},
      "interaction":{"hover":true,"navigationButtons":true,"keyboard":true}}""")

    added_nodes = set()
    edge_added  = set()
    ET_COLOR = {"cites":"#42a5f5","cited_by":"#29b6f6","has_ipc":"#ab47bc",
                "rev_has_ipc":"#ce93d8","owns":"#66bb6a","owned_by":"#a5d6a7",
                "sued":"#ef5350","sued_by":"#ff7043"}

    cf_importance = {}
    if cf_res_json:
        for k, v in cf_res_json.items():
            cf_importance[k] = abs(v.get("delta", 0))
    max_cf = max(cf_importance.values()) if cf_importance else 1.0

    def add_node(idx):
        if idx in added_nodes: return
        added_nodes.add(idx)
        label = str(pat_list[idx])
        if idx == src_idx:
            net.add_node(idx, label=label,
                         title=f"<b>[Patent A · Plaintiff in GT]</b><br>{label}",
                         color={"background": "#ff4444", "border": "#ffdd00"},
                         size=45, font={"size": 13, "color": "white"},
                         x=0, y=-100, physics=False)
            return
        elif idx == dst_idx:
            net.add_node(idx, label=label,
                         title=f"<b>[Patent B · Defendant in GT]</b><br>{label}",
                         color={"background": "#ff8800", "border": "#ffdd00"},
                         size=45, font={"size": 13, "color": "white"},
                         x=0, y=100, physics=False)
            return
        else:
            sv = shap_json.get(str(idx), 0) if shap_json else 0
            color = "#ef5350" if sv > 0.002 else ("#42a5f5" if sv < -0.002 else "#78909c")
            size, border = 14, "#aaaaaa"
            title = f"{label}<br>Shapley: {sv:.4f}" if shap_json else label
        net.add_node(idx, label=label, title=title,
                     color={"background": color, "border": border},
                     size=size, font={"size": 11, "color": "white"})

    for et in EDGE_TYPES:
        src_t, rel, dst_t = et
        if src_t != 'patent' or dst_t != 'patent': continue
        if et not in data.edge_index_dict: continue
        eidx = data.edge_index_dict[et].numpy()
        et_key = f'{src_t[:3]}–{rel}–{dst_t[:3]}'
        cf_val = cf_importance.get(et_key, 0)
        width  = 1 + 6 * (cf_val / (max_cf + 1e-9))
        color  = "#ef5350" if rel in ("sued","sued_by") else ET_COLOR.get(rel, "#888")
        title  = f"<b>SUED</b><br>{rel}" if rel in ("sued","sued_by") else f"{rel}<br>CF: {cf_val:.4f}"
        if rel in ("sued","sued_by"): width = 4

        for qi in [src_idx, dst_idx]:
            mask = (eidx[0] == qi) | (eidx[1] == qi)
            for s, d in zip(eidx[0][mask], eidx[1][mask]):
                s, d = int(s), int(d)
                add_node(s); add_node(d)
                key = (min(s,d), max(s,d), rel)
                if key not in edge_added:
                    edge_added.add(key)
                    net.add_edge(s, d, title=title, color=color, width=width, arrows="")

    add_node(src_idx); add_node(dst_idx)
    net.add_edge(src_idx, dst_idx,
                 title=f"<b>⚡ PREDICTED LAWSUIT</b><br>P(sue) = {QUERY_P:.4f}",
                 color={"color": "#ff2200", "highlight": "#ffff00"},
                 width=10, arrows="to", label=f"P={QUERY_P:.3f}")
    return net.generate_html()

cf_json   = {k: v for k, v in cache['m3'][0].items()} if 'm3' in cache else None
shap_json = {str(k): v for k, v in cache['m6'].items()} if 'm6' in cache else None

with st.spinner("서브그래프 생성 중..."):
    graph_html = build_explanation_graph(
        QUERY_S, QUERY_D, cache_key_str=cache_key,
        cf_res_json=cf_json, shap_json=shap_json
    )
components.html(graph_html, height=620, scrolling=False)
with st.expander("읽는 법"):
    st.markdown("""
- **🔴 빨간 큰 노드** = 원고 / **🟠 주황** = 피고 / **빨간 점선** = 예측 소송 관계
- **엣지 두께** = Counterfactual 중요도 (M3 실행 시 반영)
- **노드 색** = SubgraphX Shapley value (M6 실행 시 반영, 빨강=기여 파랑=방해)
- drag·zoom·hover 모두 가능
""")


# ══════════════════════════════════════════════════════════════════════════════
# VIZ C: Attention Flow Galaxy — 레이어별 애니메이션 (항상 표시)
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("## 🌌 Attention Flow Galaxy")
st.markdown(
    "GNN 레이어(L3→L2→L1)를 역추적해 원고·피고 각각의 **실제 기여 특허**를 "
    "3D 구 형태로 순차 공개합니다.  \n"
    "각 단계에서 새로 등장하는 노드만 크게 강조되고, **Final** 뷰에서 전체 기여도 크기가 확정됩니다."
)

gx_top = st.slider("Hop별 표시 특허 수 (Top-N)", 5, 25, 12, key="gx_topn")

# ── M1 자동 계산 ─────────────────────────────────────────────────────────────
if 'm1' not in cache:
    with st.spinner("Attention 자동 계산 중 (최초 1회)..."):
        cache['m1'] = (compute_attention(QUERY_S), compute_attention(QUERY_D))

# ── Attention Flow 계산 ──────────────────────────────────────────────────────
@st.cache_data
def _both_flows(src_idx, dst_idx, top_n, _ver):
    return (compute_attention_flow(src_idx, top_per_hop=top_n),
            compute_attention_flow(dst_idx, top_per_hop=top_n))

with st.spinner("Attention Flow 역추적 중..."):
    gx_flow_s, gx_flow_d = _both_flows(QUERY_S, QUERY_D, gx_top, f"{QUERY_S}_{QUERY_D}")

# ── 노드 분류 ────────────────────────────────────────────────────────────────
_fs = {n: (h, f) for h, items in gx_flow_s.items() for n, f in items}
_fd = {n: (h, f) for h, items in gx_flow_d.items() for n, f in items}

all_gx_nodes = {}
for n, (h, f) in _fs.items():
    all_gx_nodes.setdefault(n, {'flow_s': 0., 'flow_d': 0., 'hop': 99})
    all_gx_nodes[n]['flow_s'] = f
    all_gx_nodes[n]['hop']    = min(all_gx_nodes[n]['hop'], h)
for n, (h, f) in _fd.items():
    all_gx_nodes.setdefault(n, {'flow_s': 0., 'flow_d': 0., 'hop': 99})
    all_gx_nodes[n]['flow_d'] = f
    all_gx_nodes[n]['hop']    = min(all_gx_nodes[n]['hop'], h)

# 원고·피고 제외
all_gx_nodes.pop(QUERY_S, None); all_gx_nodes.pop(QUERY_D, None)

def _gx_status(n):
    in_s = n in _fs; in_d = n in _fd
    if in_s and in_d: return 'shared'
    return 'a_only' if in_s else 'b_only'

GX_BORDER = {'shared': '#ffd700', 'a_only': '#4499ff', 'b_only': '#ff8844'}
GX_BODY   = {1: '#cc3300', 2: '#cc7700', 3: '#bb9900'}  # 진한→옅은

nodes_by_hop_gx = {h: [] for h in [1, 2, 3]}
for n, info in all_gx_nodes.items():
    h = info['hop']
    if h in nodes_by_hop_gx:
        nodes_by_hop_gx[h].append(n)

# hop별 최대 flow (크기 정규화용)
def _hop_max(h):
    vals = [all_gx_nodes[n]['flow_s'] + all_gx_nodes[n]['flow_d']
            for n in nodes_by_hop_gx.get(h, [])]
    return max(vals) if vals else 1e-9

hop_max_gx = {h: _hop_max(h) for h in [1, 2, 3]}
total_max_gx = max(
    (info['flow_s'] + info['flow_d'] for info in all_gx_nodes.values()), default=1e-9)

# ── 엣지 수집 ─────────────────────────────────────────────────────────────────
gx_node_set = set(all_gx_nodes.keys()) | {QUERY_S, QUERY_D}
gx_all_edges = []
_seen_ge = set()
for et in EDGE_TYPES:
    src_t, rel, dst_t = et
    if src_t != 'patent' or dst_t != 'patent': continue
    if et not in data.edge_index_dict: continue
    ei = data.edge_index_dict[et].numpy()
    for s, d in zip(ei[0], ei[1]):
        s, d = int(s), int(d)
        if s not in gx_node_set or d not in gx_node_set: continue
        k = (min(s,d), max(s,d), rel)
        if k in _seen_ge: continue
        _seen_ge.add(k); gx_all_edges.append((s, d, rel))

# ── Three.js 그래프 데이터 준비 ──────────────────────────────────────────────
_n1 = len(nodes_by_hop_gx[1]); _n2 = len(nodes_by_hop_gx[2]); _n3 = len(nodes_by_hop_gx[3])
_sh = sum(1 for n in all_gx_nodes if _gx_status(n) == 'shared')

gx_graph_json = {
    "plt_id":    QUERY_S,
    "def_id":    QUERY_D,
    "query_p":   float(QUERY_P),
    "plt_label": str(pat_list[QUERY_S]),
    "def_label": str(pat_list[QUERY_D]),
    "hop_max":   {str(h): float(v) for h, v in hop_max_gx.items()},
    "total_max": float(total_max_gx),
    "nodes": (
        [{"id": QUERY_S, "label": str(pat_list[QUERY_S]),
          "hop": 0, "status": "plt", "flow": 1.0, "flow_s": 1.0, "flow_d": 0.0}] +
        [{"id": QUERY_D, "label": str(pat_list[QUERY_D]),
          "hop": 0, "status": "def", "flow": 1.0, "flow_s": 0.0, "flow_d": 1.0}] +
        [{"id": n, "label": str(pat_list[n]), "hop": info["hop"],
          "status": _gx_status(n), "flow": info["flow_s"] + info["flow_d"],
          "flow_s": info["flow_s"], "flow_d": info["flow_d"]}
         for n, info in all_gx_nodes.items() if info["hop"] in (1, 2, 3)]
    ),
    "links": (
        [{"source": QUERY_S, "target": QUERY_D, "rel": "gt_lawsuit"}] +
        [{"source": s, "target": d, "rel": rel}
         for s, d, rel in gx_all_edges if {s, d} != {QUERY_S, QUERY_D}]
    ),
}
gx_json_str = json.dumps(gx_graph_json)

# ── Three.js + 3d-force-graph HTML ────────────────────────────────────────────
GX_HTML_TMPL = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:#040608;overflow:hidden;font-family:'JetBrains Mono',monospace;}
  #g{width:100%;height:820px;position:relative;}
  .panel{
    position:absolute;z-index:10;
    background:rgba(4,6,8,0.88);border:1px solid #1a2535;
    border-radius:10px;padding:12px 16px;backdrop-filter:blur(10px);
    font-size:11px;line-height:1.9;color:#b0bec5;
  }
  #stats{top:12px;left:12px;}
  #legend{top:12px;right:12px;}
  #ctrl{
    position:absolute;bottom:12px;left:50%;transform:translateX(-50%);
    z-index:10;background:rgba(4,6,8,0.9);border:1px solid #1a2535;
    border-radius:12px;padding:10px 22px;display:flex;align-items:center;gap:12px;
    backdrop-filter:blur(10px);
  }
  .btn{
    background:rgba(79,195,247,.1);border:1px solid rgba(79,195,247,.4);
    color:#4fc3f7;border-radius:7px;padding:5px 14px;
    font-size:11px;font-weight:700;cursor:pointer;
    font-family:'JetBrains Mono',monospace;transition:all .15s;
  }
  .btn:hover{background:rgba(79,195,247,.22);box-shadow:0 0 8px rgba(79,195,247,.3);}
  #sld{width:200px;accent-color:#4fc3f7;cursor:pointer;}
  #sname{font-size:11px;color:#b0bec5;min-width:160px;text-align:center;}
  .hdr{font-size:10px;color:#546e7a;letter-spacing:.1em;text-transform:uppercase;margin-bottom:3px;}
  .val{font-weight:700;}
  .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:middle;}
</style>
</head>
<body>
<div id="g">
  <div class="panel" id="stats">
    <div class="hdr">현재 단계</div>
    <div id="s-step" class="val" style="color:#4fc3f7;font-size:13px;">🌑 Init</div>
    <div style="margin-top:6px;" class="hdr">공개 특허</div>
    <div id="s-rev" class="val" style="color:#e8eaf0;">0개</div>
    <div class="hdr">Bridge</div>
    <div id="s-br" class="val" style="color:#ffd700;">-</div>
    <div class="hdr">P(sue)</div>
    <div id="s-p" class="val" style="color:#ef5350;">-</div>
  </div>
  <div class="panel" id="legend">
    <div class="hdr" style="margin-bottom:4px;">노드 소속</div>
    <div><span class="dot" style="background:#ffee00;box-shadow:0 0 5px #ffee00aa;"></span>★ Plaintiff</div>
    <div><span class="dot" style="background:#ff7700;box-shadow:0 0 5px #ff7700aa;"></span>◆ Defendant</div>
    <div><span class="dot" style="background:#cc3300;border:2px solid #ffd700;"></span><span style="color:#ffd700;">Bridge (공통)</span></div>
    <div><span class="dot" style="background:#cc3300;border:2px solid #4499ff;"></span><span style="color:#4499ff;">PLT-side</span></div>
    <div><span class="dot" style="background:#cc3300;border:2px solid #ff8844;"></span><span style="color:#ff8844;">DEF-side</span></div>
    <hr style="border-color:#1a2535;margin:6px 0;">
    <div class="hdr" style="margin-bottom:4px;">Hop (진함→옅음)</div>
    <div><span class="dot" style="background:#cc3300;"></span>1-hop</div>
    <div><span class="dot" style="background:#cc7700;"></span>2-hop</div>
    <div><span class="dot" style="background:#bb9900;"></span>3-hop</div>
  </div>
  <div id="ctrl">
    <button class="btn" id="bprev">◀ Prev</button>
    <button class="btn" id="bplay">▶ Play</button>
    <button class="btn" id="bnext">Next ▶</button>
    <input type="range" id="sld" min="0" max="4" value="0" step="1">
    <div id="sname">🌑 Init</div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.138.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/3d-force-graph@1/dist/3d-force-graph.min.js"></script>
<script>
const DATA = __GRAPH_DATA__;
const STEPS = ["🌑 Init","⚡ 1-hop (L3)","🔥 2-hop (L2)","🌌 3-hop (L1)","✨ Final"];
const HOP_COLOR = {1:0xcc3300,2:0xcc7700,3:0xbb9900};
const GLOW_COLOR = {shared:0xffd700,a_only:0x4499ff,b_only:0xff8844};
const GRAY = 0x1a2530;

let step = 0, playTimer = null;

const hopMax = {};
[1,2,3].forEach(h=>{
  const v = DATA.nodes.filter(n=>n.hop===h).map(n=>n.flow);
  hopMax[h] = v.length ? Math.max(...v) : 1e-9;
});
const totalMax = Math.max(...DATA.nodes.filter(n=>n.hop>0).map(n=>n.flow), 1e-9);

const nodeById = {};
DATA.nodes.forEach(n => nodeById[n.id] = n);

function revealed(n,s){
  if(n.status==='plt'||n.status==='def') return true;
  if(s===0) return false;
  if(s===4) return true;
  return n.hop <= s;
}

function nodeRadius(n,s){
  if(n.status==='plt'||n.status==='def') return 6;
  if(!revealed(n,s)) return 1.1;
  if(s===4) return 2.5 + 6*(n.flow/totalMax);
  if(n.hop===s) return 3.5 + 7*(n.flow/(hopMax[n.hop]||1e-9));
  return 2.5;
}
function bodyColor(n,s){
  if(n.status==='plt') return 0xffee00;
  if(n.status==='def') return 0xff7700;
  if(!revealed(n,s)) return GRAY;
  return HOP_COLOR[n.hop] || 0x888888;
}
function glowColor(n,s){
  if(!revealed(n,s)) return null;
  if(n.status==='plt'||n.status==='def') return 0xffffff;
  return GLOW_COLOR[n.status] || null;
}

function makeObj(n){
  const s = step;
  const r = nodeRadius(n,s);
  const bc = bodyColor(n,s);
  const gc = glowColor(n,s);
  const grp = new THREE.Group();
  // body
  const geo = new THREE.SphereGeometry(r,16,16);
  const mat = new THREE.MeshLambertMaterial({color:bc});
  grp.add(new THREE.Mesh(geo,mat));
  // glow halo (status / GT indicator)
  if(gc !== null){
    const glowOpacity = (n.status==='plt'||n.status==='def') ? 0.22
                       : (n.hop===s||s===4) ? 0.38 : 0.15;
    const gg = new THREE.SphereGeometry(r*1.55,12,12);
    const gm = new THREE.MeshLambertMaterial({
      color:gc, transparent:true, opacity:glowOpacity, side:THREE.BackSide
    });
    grp.add(new THREE.Mesh(gg,gm));
  }
  return grp;
}

function linkCol(l){
  if(l.rel==='gt_lawsuit') return '#ff3300';
  const sid = typeof l.source==='object' ? l.source.id : l.source;
  const did = typeof l.target==='object' ? l.target.id : l.target;
  const src = nodeById[sid], dst = nodeById[did];
  if(!src||!dst) return '#0c1520';
  if(step===0) return '#0c1520';
  if(step===4) return '#225588';
  const mh = Math.max(src.hop||0, dst.hop||0);
  return mh <= step ? '#225588' : '#0c1520';
}
function linkW(l){ return l.rel==='gt_lawsuit' ? 2.5 : 0.4; }
function linkOp(l){ return l.rel==='gt_lawsuit' ? 0.9 : 0.25; }

const Graph = ForceGraph3D()(document.getElementById('g'))
  .width(document.getElementById('g').clientWidth)
  .height(820)
  .backgroundColor('#040608')
  .graphData({nodes:DATA.nodes.map(n=>({...n})), links:DATA.links.map(l=>({...l}))})
  .nodeId('id')
  .nodeLabel(n=>`<div style="font:11px 'JetBrains Mono',monospace;
    background:#040608dd;padding:7px 11px;border-radius:7px;
    border:1px solid #334;color:#e8eaf0;line-height:1.7;">
    <b style="color:#4fc3f7;">${n.label}</b><br>
    Hop:${n.hop} &nbsp; Status:<span style="color:${{shared:'#ffd700',a_only:'#4499ff',b_only:'#ff8844',plt:'#ffee00',def:'#ff7700'}[n.status]||'#aaa'}">${n.status}</span><br>
    Flow:${n.flow?n.flow.toFixed(4):'-'}
    ${n.flow_s?`<br>PLT:${n.flow_s.toFixed(4)} &nbsp; DEF:${n.flow_d.toFixed(4)}`:''}
  </div>`)
  .nodeThreeObject(n=>makeObj(n))
  .nodeThreeObjectExtend(false)
  .linkColor(l=>linkCol(l))
  .linkWidth(l=>linkW(l))
  .linkOpacity(l=>linkOp(l))
  .linkDirectionalParticles(l=>l.rel==='gt_lawsuit'?5:0)
  .linkDirectionalParticleColor(()=>'#ff4400')
  .linkDirectionalParticleWidth(2)
  .linkDirectionalParticleSpeed(0.005)
  .showNavInfo(false)
  .enableNodeDrag(true);

// Physics
Graph.d3Force('charge').strength(-130);
Graph.d3Force('link')
  .distance(l=>{
    const sid = typeof l.source==='object'?l.source.id:l.source;
    const did = typeof l.target==='object'?l.target.id:l.target;
    const s2=nodeById[sid], d2=nodeById[did];
    if(!s2||!d2) return 50;
    if(l.rel==='gt_lawsuit') return 25;
    return 25 + Math.max(s2.hop||1,d2.hop||1)*22;
  }).strength(0.28);
Graph.d3Force('center').strength(0.05);

// Star field
setTimeout(()=>{
  const scene = Graph.scene();
  const geo = new THREE.BufferGeometry();
  const pos = [];
  for(let i=0;i<3500;i++){
    pos.push((Math.random()-.5)*3500,(Math.random()-.5)*3500,(Math.random()-.5)*3500);
  }
  geo.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  const mat = new THREE.PointsMaterial({
    color:0xffffff,size:1.3,sizeAttenuation:true,transparent:true,opacity:0.55
  });
  scene.add(new THREE.Points(geo,mat));
}, 400);

// Stats
function updateStats(){
  const rev = DATA.nodes.filter(n=>revealed(n,step)&&n.hop>0).length;
  const br  = DATA.nodes.filter(n=>revealed(n,step)&&n.status==='shared').length;
  document.getElementById('s-step').textContent = STEPS[step];
  document.getElementById('s-rev').textContent  = rev+'개';
  document.getElementById('s-br').textContent   = br+'개';
  document.getElementById('s-p').textContent    = DATA.query_p.toFixed(4);
  document.getElementById('sname').textContent  = STEPS[step];
  document.getElementById('sld').value = step;
}

function setStep(s){
  step = Math.max(0,Math.min(4,s));
  Graph.nodeThreeObject(n=>makeObj(n));
  Graph.linkColor(l=>linkCol(l));
  updateStats();
}

function stopPlay(){
  if(playTimer){clearInterval(playTimer);playTimer=null;}
  document.getElementById('bplay').textContent='▶ Play';
}

document.getElementById('bprev').onclick = ()=>{stopPlay();setStep(step-1);};
document.getElementById('bnext').onclick = ()=>{stopPlay();setStep(step+1);};
document.getElementById('sld').oninput  = e=>{stopPlay();setStep(+e.target.value);};
document.getElementById('bplay').onclick = ()=>{
  if(playTimer){stopPlay();return;}
  document.getElementById('bplay').textContent='⏸ Pause';
  playTimer=setInterval(()=>{
    const nx=step+1;
    if(nx>4){stopPlay();return;}
    setStep(nx);
  },2300);
};

updateStats();
setTimeout(()=>Graph.d3AlphaDecay(0.03),5000);
</script>
</body>
</html>"""

GX_HTML = GX_HTML_TMPL.replace('__GRAPH_DATA__', gx_json_str)

# ── 렌더링: 3D Galaxy + 우측 기여 특허 표 ─────────────────────────────────────
gx_main_col, gx_tbl_col = st.columns([3, 1])

with gx_main_col:
    components.html(GX_HTML, height=860, scrolling=False)

with gx_tbl_col:
    st.markdown("#### 📋 기여 특허 목록")

    bridge_nodes = sorted(
        [(n, all_gx_nodes[n]['flow_s']+all_gx_nodes[n]['flow_d'])
         for n in all_gx_nodes if _gx_status(n)=='shared'], key=lambda x:-x[1])
    st.markdown("🟡 **Bridge (공통)**")
    if bridge_nodes:
        for n, f in bridge_nodes[:8]:
            st.markdown(
                f"<small>`{str(pat_list[n])[:16]}`&nbsp;"
                f"<span style='color:#ffd700'>{f:.3f}</span></small>",
                unsafe_allow_html=True)
    else:
        st.caption("공통 특허 없음")

    st.divider()
    plt_nodes = sorted(
        [(n, all_gx_nodes[n]['flow_s']) for n in all_gx_nodes if _gx_status(n)=='a_only'],
        key=lambda x:-x[1])
    st.markdown("🔵 **PLT-side**")
    for n, f in plt_nodes[:6]:
        st.markdown(
            f"<small>`{str(pat_list[n])[:16]}`&nbsp;"
            f"<span style='color:#4499ff'>{f:.3f}</span></small>",
            unsafe_allow_html=True)

    st.divider()
    def_nodes = sorted(
        [(n, all_gx_nodes[n]['flow_d']) for n in all_gx_nodes if _gx_status(n)=='b_only'],
        key=lambda x:-x[1])
    st.markdown("🟠 **DEF-side**")
    for n, f in def_nodes[:6]:
        st.markdown(
            f"<small>`{str(pat_list[n])[:16]}`&nbsp;"
            f"<span style='color:#ff8844'>{f:.3f}</span></small>",
            unsafe_allow_html=True)

    st.divider()
    st.markdown(
        f"<small>1-hop:**{_n1}** 2-hop:**{_n2}** 3-hop:**{_n3}** Bridge:**{_sh}**</small>",
        unsafe_allow_html=True)

with st.expander("읽는 법"):
    st.markdown("""
**▶ Play** 또는 슬라이더로 단계를 탐색합니다. 노드를 드래그해 회전·확대 가능.

| 단계 | 내용 |
|------|------|
| 🌑 Init | 원고★·피고◆만 표시, 나머지 회색 |
| ⚡ 1-hop | L3 역추적 — 직접 기여 특허 등장 (크게 강조) |
| 🔥 2-hop | L2 역추적 — 간접 기여 특허 추가 |
| 🌌 3-hop | L1 역추적 — 원거리 기여 특허 추가 |
| ✨ Final | 전체 Attention Flow 기반 최종 크기 확정 |

**노드 halo 색** (소속): 🟡 Bridge(공통) &nbsp; 🔵 PLT-side &nbsp; 🟠 DEF-side
**노드 body 색**: 진함(1-hop) → 중간(2-hop) → 옅음(3-hop)
**GT 소송 엣지**: 빨간 실선 + 파티클
**물리 시뮬레이션**: 노드끼리 서로 밀고 당겨 자동 배치됩니다
""")


# ── XAI 방법별 분석 (버튼 클릭 후) ─────────────────────────────────────────
st.divider()
st.markdown("## 🔎 XAI 방법별 분석")
if not run_btn:
    st.info("사이드바에서 방법을 선택한 후 **🚀 분석 실행**을 클릭하세요.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# M1: Attention Visualization
# ══════════════════════════════════════════════════════════════════════════════
if run_m1:
    st.markdown('<div class="method-header">M1. Attention Visualization in GAT</div>', unsafe_allow_html=True)
    st.markdown('<div class="desc-box">각 GAT 레이어에서 edge type별 attention weight 추출<br>'
                '→ 어떤 이웃 노드가 임베딩 형성에 가장 많이 기여했는지 시각화</div>', unsafe_allow_html=True)

    if 'm1' not in cache:
        with st.spinner("Attention 계산 중..."):
            cache['m1'] = (compute_attention(QUERY_S), compute_attention(QUERY_D))

    attn_src, attn_dst = cache['m1']
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), facecolor='#0e1117')
    fig.suptitle(f'Attention: {pat_list[QUERY_S]} -> {pat_list[QUERY_D]}',
                 fontsize=12, color='white')

    for row_i, (node_attn, node_label_str) in enumerate([
        (attn_src, f'[PLT] {pat_list[QUERY_S]}'),
        (attn_dst, f'[DEF] {pat_list[QUERY_D]}')
    ]):
        for col_i, layer in enumerate(['L1', 'L2', 'L3']):
            ax = axes[row_i][col_i]
            ax.set_facecolor('#1e2130')
            ax.set_title(f'{node_label_str[:20]} | {layer}', fontsize=8, color='white')
            ax.tick_params(colors='white', labelsize=7)
            for spine in ax.spines.values(): spine.set_edgecolor('#444')

            bars = []
            for et, v in node_attn.get(layer, {}).items():
                src_t, rel, dst_t = et
                for sn, av in zip(v['src'], v['attn']):
                    lbl = node_label(sn, src_t)[:14]
                    bars.append((lbl, av, rel))
            if not bars:
                ax.text(0.5, 0.5, 'No incoming edges', ha='center', va='center', color='gray')
                ax.axis('off'); continue
            bars.sort(key=lambda x: -x[1]); bars = bars[:8]
            lbls_b = [b[0] for b in bars]; vals_b = [b[1] for b in bars]
            rels_b = [b[2] for b in bars]
            urels = list(dict.fromkeys(rels_b))
            cmap = {r: f'C{i2}' for i2, r in enumerate(urels)}
            ax.barh(range(len(bars)), vals_b, color=[cmap[r] for r in rels_b], alpha=0.85)
            ax.set_yticks(range(len(bars))); ax.set_yticklabels(lbls_b, color='white', fontsize=6)
            ax.set_xlabel('Attention', color='gray', fontsize=7)
            ax.legend(handles=[Patch(color=cmap[r], label=r) for r in urels],
                      fontsize=5, facecolor='#1e2130', labelcolor='white')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# M2: Grad-CAM
# ══════════════════════════════════════════════════════════════════════════════
if run_m2:
    st.markdown('<div class="method-header">M2. Grad-CAM for Link Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="desc-box">score = sigmoid(h_src · h_dst) 에 대한 gradient × activation<br>'
                '→ 레이어별로 어떤 피처 차원이 소송 예측에 기여했는지 시각화</div>', unsafe_allow_html=True)

    def compute_grad_cam(src_idx, dst_idx):
        model.eval()
        x_pat_raw = torch.cat([data['patent'].struct, data['patent'].x], dim=1)
        x = {
            'patent':  model.pat_proj(x_pat_raw),
            'company': F.leaky_relu(model.co_proj(data['company'].x)),
            'ipc':     model.ipc_emb.weight,
        }
        acts_fns = [F.leaky_relu, F.leaky_relu, torch.tanh]
        intermediate = []
        for li, (conv, bn, act) in enumerate(zip(model.convs[:3], model.bns, acts_fns)):
            x_new  = conv(x, data.edge_index_dict)
            x_next = {}
            for k, v in x_new.items():
                activated = model.drop(act(bn[k](v.view(-1, model.hid)))) if k in bn \
                            else model.drop(act(v))
                if k == 'patent':
                    activated.retain_grad()   # 비-leaf 텐서의 grad 보존
                    intermediate.append(activated)
                x_next[k] = activated
            x = x_next
        x_out = model.convs[3](x, data.edge_index_dict)
        h     = {k: v.view(-1, model.out) for k, v in x_out.items()}
        score = torch.sigmoid((h['patent'][src_idx] * h['patent'][dst_idx]).sum())
        score.backward()
        results = {}
        for li, act_t in enumerate(intermediate):
            if act_t.grad is None:
                continue
            for ni, tag in [(src_idx, 'src'), (dst_idx, 'dst')]:
                g   = act_t.grad[ni].detach().cpu()
                a   = act_t[ni].detach().cpu()
                cam = F.relu(g * a).numpy()
                results.setdefault(f'Layer {li+1}', {})[tag] = cam
        return results, score.item()

    if 'm2' not in cache:
        with st.spinner("Grad-CAM 계산 중..."):
            cache['m2'] = compute_grad_cam(QUERY_S, QUERY_D)

    grad_res, pred_score = cache['m2']
    n_layers = len(grad_res)

    if n_layers == 0:
        st.warning("Grad-CAM: 그래디언트가 추출되지 않았습니다. 다른 쌍을 선택하거나 재실행하세요.")
        cache.pop('m2', None)
    else:
        fig, axes = plt.subplots(2, n_layers, figsize=(5*n_layers, 7), facecolor='#0e1117')
        if n_layers == 1:
            axes = [[axes[0]], [axes[1]]]
        fig.suptitle(f'Grad-CAM | P(sue)={pred_score:.3f}', fontsize=12, color='white')
        for col_i, (layer_name, vals) in enumerate(grad_res.items()):
            for row_i, (tag, node_lbl) in enumerate([
                ('src', f'[PLT] {pat_list[QUERY_S]}'),
                ('dst', f'[DEF] {pat_list[QUERY_D]}')
            ]):
                ax = axes[row_i][col_i]
                ax.set_facecolor('#1e2130')
                ax.tick_params(colors='white', labelsize=7)
                for spine in ax.spines.values(): spine.set_edgecolor('#444')
                cam = vals.get(tag, np.zeros(1))
                top_idx = np.argsort(-cam)[:20]
                ax.bar(range(min(20, len(top_idx))), cam[top_idx[:20]], color='#4fc3f7', alpha=0.8)
                ax.set_title(f'{layer_name} | {node_lbl[:18]}', fontsize=8, color='white')
                ax.set_xlabel('Feature dim (top-20)', color='gray', fontsize=7)
                ax.set_ylabel('Grad-CAM', color='gray', fontsize=7)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        cols = st.columns(n_layers)
        for col_i, (layer_name, vals) in enumerate(grad_res.items()):
            with cols[col_i]:
                total_s = vals.get('src', np.zeros(1)).sum()
                total_d = vals.get('dst', np.zeros(1)).sum()
                st.metric(f"{layer_name} PLT", f"{total_s:.3f}")
                st.metric(f"{layer_name} DEF", f"{total_d:.3f}")
    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# M3: Counterfactual
# ══════════════════════════════════════════════════════════════════════════════
if run_m3:
    st.markdown('<div class="method-header">M3. Counterfactual Explanations</div>', unsafe_allow_html=True)
    st.markdown('<div class="desc-box">각 edge type의 엣지를 제거했을 때 소송 예측 확률 변화<br>'
                '→ 확률 감소가 클수록 그 연결 관계가 예측에 중요함</div>', unsafe_allow_html=True)

    N_RUNS = 10   # dropout 평균 횟수

    def _bn_train_mode():
        """BatchNorm만 train 모드로 전환 (dropout은 그대로 켜짐)."""
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.train()

    def _forward_avg(n_runs=N_RUNS):
        """dropout + BN-train 모드로 n_runs 평균 forward."""
        _bn_train_mode()
        probs = []
        with torch.no_grad():
            for _ in range(n_runs):
                h = model(data)['patent']
                probs.append(float(torch.sigmoid(
                    (h[QUERY_S] * h[QUERY_D]).sum())))
        model.eval()   # 원복
        return float(sum(probs) / len(probs))

    def compute_counterfactual(src_idx, dst_idx):
        # ── 원본 확률: BN train 모드 + dropout 평균 ──────────
        model.train()
        _bn_train_mode()
        orig_p = _forward_avg()

        results = {}
        for et in EDGE_TYPES:
            src_t, rel, dst_t = et
            if et not in data.edge_index_dict: continue
            orig_edge = data.edge_index_dict[et].clone()
            keep = torch.ones(orig_edge.shape[1], dtype=torch.bool)
            if src_t == 'patent':
                keep &= (orig_edge[0] != src_idx) & (orig_edge[0] != dst_idx)
            if dst_t == 'patent':
                keep &= (orig_edge[1] != src_idx) & (orig_edge[1] != dst_idx)
            n_rem = int((~keep).sum())
            if n_rem == 0: continue

            # ── CF 확률: 엣지 제거 후 동일 방식으로 평균 ──────
            data.edge_index_dict[et] = orig_edge[:, keep]
            cf_p = _forward_avg()
            data.edge_index_dict[et] = orig_edge

            results[f'{src_t[:3]}–{rel}–{dst_t[:3]}'] = {
                'prob': cf_p, 'n_removed': n_rem, 'delta': orig_p - cf_p
            }

        model.eval()   # 완전 원복
        return results, orig_p

    if 'm3' not in cache:
        with st.spinner(f"Counterfactual 계산 중… (BN train 모드, {N_RUNS}회 평균)"):
            cache['m3'] = compute_counterfactual(QUERY_S, QUERY_D)

    cf_res, orig_p = cache['m3']
    items_cf = sorted(cf_res.items(), key=lambda x: -abs(x[1]['delta']))

    col_left, col_right = st.columns(2)
    with col_left:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0e1117')
        ax.set_facecolor('#1e2130')
        lbls  = [x[0] for x in items_cf]
        delts = [x[1]['delta'] for x in items_cf]
        ax.barh(range(len(lbls)), delts,
                color=['#ef5350' if d > 0 else '#42a5f5' for d in delts], alpha=0.85)
        ax.axvline(0, color='white', linewidth=1)
        ax.set_yticks(range(len(lbls))); ax.set_yticklabels(lbls, color='white', fontsize=8)
        ax.set_xlabel('Delta P (Original - CF)\nPositive = edge is important', color='gray', fontsize=8)
        ax.set_title('Importance by Edge Type', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_edgecolor('#444')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with col_right:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0e1117')
        ax.set_facecolor('#1e2130')
        probs_cf = [x[1]['prob'] for x in items_cf]
        nrem_cf  = [x[1]['n_removed'] for x in items_cf]
        ax.barh(range(len(lbls)), probs_cf, color='#42a5f5', alpha=0.8)
        ax.axvline(orig_p, color='#ef5350', linestyle='--', label=f'Original P={orig_p:.3f}')
        ax.set_yticks(range(len(lbls)))
        ax.set_yticklabels([f'{l} (removed {n})' for l, n in zip(lbls, nrem_cf)],
                           color='white', fontsize=7)
        ax.set_xlabel('P(sue) after removal', color='gray', fontsize=8)
        ax.set_title('P(sue) after Edge Removal', color='white')
        ax.legend(facecolor='#1e2130', labelcolor='white', fontsize=8)
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_edgecolor('#444')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    max_delta = max(abs(v['delta']) for v in cf_res.values()) if cf_res else 0
    if max_delta < 0.001:
        st.warning(
            f"⚠️ 최대 delta = {max_delta:.6f} — 엣지 제거 후에도 확률 변화가 거의 없음. "
            "이 예측은 그래프 구조보다 **입력 피처(FAISS 유사도)** 에 의해 결정될 가능성이 높습니다."
        )
    st.dataframe(
        [{'Edge Type': k, 'Removed': v['n_removed'],
          'Original P': f"{orig_p:.6f}", 'CF P': f"{v['prob']:.6f}",
          'Delta': f"{v['delta']:.6f}"}
         for k, v in items_cf],
        use_container_width=True, height=280
    )
    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# M4: GNNExplainer
# ══════════════════════════════════════════════════════════════════════════════
if run_m4:
    st.markdown('<div class="method-header">M4. GNNExplainer (Link Prediction)</div>', unsafe_allow_html=True)
    st.markdown('<div class="desc-box">예측 확률을 최대화하는 edge mask를 학습합니다 (Straight-Through Estimator)<br>'
                '→ mask prob이 높을수록 예측에 필수적인 연결</div>', unsafe_allow_html=True)

    def compute_gnn_explainer(src_idx, dst_idx, n_epochs=300):
        local_masks = {}; params = []; orig_edges = {}
        for et in EDGE_TYPES:
            src_t, rel, dst_t = et
            if et not in data.edge_index_dict: continue
            eidx = data.edge_index_dict[et]; orig_edges[et] = eidx.clone()
            rel_mask = torch.zeros(eidx.shape[1], dtype=torch.bool)
            if src_t == 'patent':
                rel_mask |= (eidx[0] == src_idx) | (eidx[0] == dst_idx)
            if dst_t == 'patent':
                rel_mask |= (eidx[1] == src_idx) | (eidx[1] == dst_idx)
            n_rel = int(rel_mask.sum())
            if n_rel == 0: continue
            mask_logit = nn.Parameter(torch.zeros(n_rel))
            params.append(mask_logit); local_masks[et] = (rel_mask, mask_logit)
        if not params: return {}
        optimizer = torch.optim.Adam(params, lr=0.05)
        for _ in range(n_epochs):
            optimizer.zero_grad()
            for et, (rel_mask, mask_logit) in local_masks.items():
                eidx = orig_edges[et]; soft = torch.sigmoid(mask_logit)
                hard = (torch.rand_like(soft) < soft.detach()).float()
                full_keep = torch.ones(eidx.shape[1], dtype=torch.bool)
                full_keep[rel_mask] = hard.bool()
                data.edge_index_dict[et] = eidx[:, full_keep]
            h_m   = model(data)['patent']
            score = torch.sigmoid((h_m[src_idx] * h_m[dst_idx]).sum())
            for et in local_masks: data.edge_index_dict[et] = orig_edges[et]
            size_loss = sum(torch.sigmoid(ml).mean() for _, (_, ml) in local_masks.items())
            (-torch.log(score + 1e-6) + 5e-4 * size_loss).backward()
            optimizer.step()
        results = {}
        for et, (rel_mask, mask_logit) in local_masks.items():
            eidx = orig_edges[et]; _, rel_name, dst_t_str = et
            src_t_str = et[0]
            results[et] = {
                'src_nodes': eidx[0][rel_mask].numpy(),
                'dst_nodes': eidx[1][rel_mask].numpy(),
                'mask_probs': torch.sigmoid(mask_logit).detach().numpy(),
                'rel': rel_name, 'src_t': src_t_str, 'dst_t': dst_t_str
            }
        return results

    if 'm4' not in cache:
        with st.spinner("GNNExplainer 학습 중 (~30초)..."):
            cache['m4'] = compute_gnn_explainer(QUERY_S, QUERY_D)

    gnn_res = cache['m4']
    if gnn_res:
        n_et = len(gnn_res)
        fig, axes = plt.subplots(1, n_et, figsize=(max(6, 5*n_et), 6), facecolor='#0e1117')
        if n_et == 1: axes = [axes]
        fig.suptitle('GNNExplainer: Key Edges by Mask Probability', color='white', fontsize=12)
        for ax_i, (et, v) in enumerate(gnn_res.items()):
            ax = axes[ax_i]; ax.set_facecolor('#16192b')
            ax.tick_params(colors='white', labelsize=7)
            for spine in ax.spines.values(): spine.set_edgecolor('#444')
            elabels = [f'{node_label(int(s), v["src_t"])[:10]}→{node_label(int(d), v["dst_t"])[:10]}'
                       for s, d in zip(v['src_nodes'], v['dst_nodes'])]
            sidx = np.argsort(-v['mask_probs'])[:10]
            mp_top = v['mask_probs'][sidx]
            cmap = plt.cm.RdYlGn
            bar_colors = [cmap(float(p)) for p in mp_top]
            ax.barh(range(len(sidx)), mp_top, color=bar_colors, alpha=0.9)
            ax.axvline(0.5, color='#ef5350', linestyle='--', alpha=0.8, label='threshold (0.5)')
            ax.set_yticks(range(len(sidx)))
            ax.set_yticklabels([elabels[i2] for i2 in sidx], color='white', fontsize=7)
            ax.set_xlabel('Mask probability', color='gray', fontsize=7)
            ax.set_title(v['rel'], color='white', fontsize=9)
            ax.legend(fontsize=7, facecolor='#1e2130', labelcolor='white')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# M5: PGExplainer
# ══════════════════════════════════════════════════════════════════════════════
if run_m5:
    st.markdown('<div class="method-header">M5. PGExplainer</div>', unsafe_allow_html=True)
    st.markdown('<div class="desc-box">MLP(h_i ‖ h_j) → [0,1] 를 학습해 이웃 노드 중요도를 파라미터화합니다<br>'
                '→ 원고/피고 주변에서 소송 예측에 기여하는 특허를 식별</div>', unsafe_allow_html=True)

    class PGEdgeMLP(nn.Module):
        def __init__(self, emb_dim=64, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2*emb_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden//2), nn.ReLU(),
                nn.Linear(hidden//2, 1))
        def forward(self, hi, hj):
            return self.net(torch.cat([hi, hj], dim=-1)).squeeze(-1)

    def compute_pg_explainer(src_idx, dst_idx, n_epochs=300):
        h_t   = torch.tensor(h_np, dtype=torch.float32)
        pg    = PGEdgeMLP(emb_dim=h_np.shape[1])
        opt   = torch.optim.Adam(pg.parameters(), lr=1e-3)
        rng   = np.random.default_rng(42)
        pos_s = torch.tensor(sued_e[0], dtype=torch.long)
        pos_d = torch.tensor(sued_e[1], dtype=torch.long)
        for _ in range(n_epochs):
            opt.zero_grad()
            ns = torch.tensor(rng.integers(0, len(h_np), size=len(pos_s)), dtype=torch.long)
            nd = torch.tensor(rng.integers(0, len(h_np), size=len(pos_s)), dtype=torch.long)
            all_s = torch.cat([pos_s, ns]); all_d = torch.cat([pos_d, nd])
            lbl   = torch.cat([torch.ones(len(pos_s)), torch.zeros(len(ns))])
            loss  = F.binary_cross_entropy_with_logits(pg(h_t[all_s], h_t[all_d]), lbl)
            loss.backward(); opt.step()
        pg.eval()
        src_nb = get_patent_neighbors(src_idx)
        dst_nb = get_patent_neighbors(dst_idx)
        with torch.no_grad():
            src_scores = {nb: float(torch.sigmoid(pg(h_t[src_idx].unsqueeze(0), h_t[nb].unsqueeze(0))))
                          for nb in src_nb}
            dst_scores = {nb: float(torch.sigmoid(pg(h_t[dst_idx].unsqueeze(0), h_t[nb].unsqueeze(0))))
                          for nb in dst_nb}
        return src_scores, dst_scores

    if 'm5' not in cache:
        with st.spinner("PGExplainer 학습 중 (~20초)..."):
            cache['m5'] = compute_pg_explainer(QUERY_S, QUERY_D)

    src_scores, dst_scores = cache['m5']
    col_left, col_right = st.columns(2)

    for col, scores, node_lbl, bar_color, role in [
        (col_left, src_scores, f'Plaintiff: {pat_list[QUERY_S]}', '#42a5f5', 'PLT'),
        (col_right, dst_scores, f'Defendant: {pat_list[QUERY_D]}', '#ef5350', 'DEF')
    ]:
        with col:
            st.caption(node_lbl)
            if scores:
                top = sorted(scores.items(), key=lambda x: -x[1])[:10]
                fig, ax = plt.subplots(figsize=(7, 5), facecolor='#0e1117')
                ax.set_facecolor('#1e2130')
                ax.tick_params(colors='white', labelsize=8)
                for spine in ax.spines.values(): spine.set_edgecolor('#444')
                lbls_pg = [str(pat_list[n])[:18] for n, _ in top]
                vals_pg = [v for _, v in top]
                ax.barh(range(len(top)), vals_pg, color=bar_color, alpha=0.8)
                ax.set_yticks(range(len(top))); ax.set_yticklabels(lbls_pg, color='white', fontsize=7)
                ax.set_xlabel('PG Importance Score', color='gray', fontsize=8)
                ax.set_title(f'[{role}] Neighbor Patent Importance', color='white', fontsize=9)
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# M6: SubgraphX
# ══════════════════════════════════════════════════════════════════════════════
if run_m6:
    st.markdown('<div class="method-header">M6. SubgraphX (Monte Carlo Shapley)</div>', unsafe_allow_html=True)
    st.markdown('<div class="desc-box">이웃 노드들의 Shapley value를 Monte Carlo 샘플링으로 추정합니다<br>'
                '→ 양수: 소송 예측에 기여 / 음수: 예측을 방해하는 노드</div>', unsafe_allow_html=True)

    def compute_subgraphx(src_idx, dst_idx, n_samples=60):
        def pred_masked(masked_nodes):
            orig_x   = data['patent'].x.clone()
            orig_str = data['patent'].struct.clone()
            for n in masked_nodes:
                data['patent'].x[n] = 0.0; data['patent'].struct[n] = 0.0
            with torch.no_grad():
                h = model(data)['patent']
                sc = float(torch.sigmoid((h[src_idx] * h[dst_idx]).sum()))
            data['patent'].x = orig_x; data['patent'].struct = orig_str
            return sc
        src_nb = get_patent_neighbors(src_idx)
        dst_nb = get_patent_neighbors(dst_idx)
        cands  = list((src_nb | dst_nb) - {src_idx, dst_idx})[:25]
        rng    = np.random.default_rng(42)
        shapley = defaultdict(float); counts = defaultdict(int)
        for _ in range(n_samples):
            perm = rng.permutation(cands); S = set()
            f_S  = pred_masked(set(cands) - S)
            for v in perm:
                S.add(v)
                f_Sv = pred_masked(set(cands) - S)
                shapley[v] += f_Sv - f_S; counts[v] += 1; f_S = f_Sv
        return {v: shapley[v] / counts[v] for v in cands}

    if 'm6' not in cache:
        with st.spinner("SubgraphX Shapley 계산 중 (~1-2분)..."):
            cache['m6'] = compute_subgraphx(QUERY_S, QUERY_D)

    shap_vals = cache['m6']
    sorted_sv = sorted(shap_vals.items(), key=lambda x: -abs(x[1]))[:15]
    fig, ax   = plt.subplots(figsize=(10, 7), facecolor='#0e1117')
    ax.set_facecolor('#1e2130')
    ax.tick_params(colors='white', labelsize=9)
    for spine in ax.spines.values(): spine.set_edgecolor('#444')
    lbls_sv  = [str(pat_list[n])[:20] for n, _ in sorted_sv]
    vals_sv  = [v for _, v in sorted_sv]
    ax.barh(range(len(sorted_sv)), vals_sv,
            color=['#ef5350' if v > 0 else '#42a5f5' for v in vals_sv], alpha=0.85)
    ax.axvline(0, color='white', linewidth=1)
    ax.set_yticks(range(len(sorted_sv))); ax.set_yticklabels(lbls_sv, color='white', fontsize=9)
    ax.set_xlabel('Shapley Value  (+: helpful,  -: harmful)', color='gray', fontsize=9)
    ax.set_title('Shapley Value by Neighbor Node', color='white', fontsize=12)
    ax.legend(handles=[Patch(color='#ef5350', label='Helpful (+)'),
                        Patch(color='#42a5f5', label='Harmful (-)')],
              facecolor='#1e2130', labelcolor='white')
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# M7: GraphSHAP
# ══════════════════════════════════════════════════════════════════════════════
if run_m7:
    st.markdown('<div class="method-header">M7. GraphSHAP (KernelSHAP Feature Attribution)</div>', unsafe_allow_html=True)
    st.markdown('<div class="desc-box">입력 피처를 그룹으로 나눠 각 그룹의 Shapley value를 KernelSHAP으로 추정합니다<br>'
                '→ struct(3d) vs FAISS 임베딩 구간별로 소송 예측 기여도를 분석</div>', unsafe_allow_html=True)

    def compute_graphshap(src_idx, dst_idx, n_groups=15, n_samples=200):
        faiss_dim = data['patent'].x.shape[1]
        chunk = faiss_dim // (n_groups - 1)
        groups = [('struct', None, 'struct')]
        for g in range(n_groups - 1):
            s = g * chunk; e = min(s + chunk, faiss_dim)
            groups.append((f'FAISS_{s}-{e}', (s, e), 'faiss'))
        M = len(groups)

        def shap_for_node(target_idx, other_idx):
            orig_x   = data['patent'].x[target_idx].clone()
            orig_str = data['patent'].struct[target_idx].clone()
            rng = np.random.default_rng(42)
            X_s, Y_s, W_s = [], [], []
            for _ in range(n_samples):
                inc = rng.random(M) > 0.5
                data['patent'].x[target_idx]      = orig_x.clone()
                data['patent'].struct[target_idx]  = orig_str.clone()
                for gi, (gname, grange, gtype) in enumerate(groups):
                    if not inc[gi]:
                        if gtype == 'struct': data['patent'].struct[target_idx] = 0.0
                        else:
                            gs, ge = grange
                            data['patent'].x[target_idx, gs:ge] = 0.0
                with torch.no_grad():
                    hm = model(data)['patent']
                    sc = float(torch.sigmoid((hm[target_idx] * hm[other_idx]).sum()))
                z = max(1, min(int(inc.sum()), M-1))
                w = (M - 1) / (sp_comb(M, z) * z * (M - z) + 1e-10)
                X_s.append(inc.astype(float)); Y_s.append(sc); W_s.append(w)
            data['patent'].x[target_idx]     = orig_x
            data['patent'].struct[target_idx] = orig_str
            X = np.array(X_s); Y = np.array(Y_s); W = np.array(W_s)
            Wm = np.diag(W); A = X.T @ Wm @ X + 1e-8 * np.eye(M); b = X.T @ Wm @ Y
            phi = np.linalg.solve(A, b)
            return {groups[gi][0]: float(phi[gi]) for gi in range(M)}

        return shap_for_node(src_idx, dst_idx), shap_for_node(dst_idx, src_idx)

    if 'm7' not in cache:
        with st.spinner("GraphSHAP KernelSHAP 계산 중 (~1분)..."):
            cache['m7'] = compute_graphshap(QUERY_S, QUERY_D)

    shap_src, shap_dst = cache['m7']
    col_left, col_right = st.columns(2)

    for col, sv, node_lbl, role in [
        (col_left, shap_src, f'{pat_list[QUERY_S]}', 'PLT'),
        (col_right, shap_dst, f'{pat_list[QUERY_D]}', 'DEF')
    ]:
        with col:
            slist = sorted(sv.items(), key=lambda x: -abs(x[1]))
            fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0e1117')
            ax.set_facecolor('#1e2130')
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values(): spine.set_edgecolor('#444')
            gnames = [g[0] for g in slist]; gvals = [g[1] for g in slist]
            ax.barh(range(len(gnames)), gvals,
                    color=['#4caf50' if v > 0 else '#ef5350' for v in gvals], alpha=0.85)
            ax.axvline(0, color='white', linewidth=1)
            ax.set_yticks(range(len(gnames))); ax.set_yticklabels(gnames, color='white', fontsize=8)
            ax.set_xlabel('SHAP Value (contribution)', color='gray', fontsize=8)
            ax.set_title(f'[{role}] {node_lbl[:18]}\nFeature Group Importance', color='white', fontsize=10)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## All Test Pair Predictions")
rows = []
for i, (s, d, p) in enumerate(test_pairs):
    rows.append({
        '#': i+1,
        'Plaintiff': str(pat_list[s]),
        'Defendant': str(pat_list[d]),
        'P(sue)': round(p, 4),
        'Pred': 'O' if p >= best_thr else 'X'
    })
st.dataframe(rows, use_container_width=True, height=400)

