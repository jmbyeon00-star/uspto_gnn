# 2026-07-13 작업 정리

## 목표

기존 `xai_compute.py` + `xai_viz_app.py` (monolith 단일파일 구조)를 **Compute / Viewer 분리 아키텍처**로 리팩터링하고, Streamlit 멀티페이지 대시보드로 전환.

---

## 아키텍처 개요

```
HeteroGAT 모델 (lib/ → viz_app/backend/)
        ↓
compute/ 파이프라인 → results/*.json
        ↓
Streamlit 멀티페이지 뷰어 (app.py + pages/)
```

---

## 파일 구조

```
src/data_266/0713/
├── app.py                    ← 홈(대시보드 진입점) + 사이드바
├── config.py                 ← 색상 팔레트, 테마, 전역 상수
├── xai_compute.py            ← 구버전 monolith compute (v3, 참조용)
├── xai_viz_app.py            ← 구버전 monolith viewer (v3, 참조용)
│
├── compute/                  ← 단계별 파이프라인 (JSON 생성)
│   ├── common.py
│   ├── run_all.py            ← 전체 파이프라인 실행 진입점
│   ├── 00_graph.py
│   ├── 00b_full_graph.py
│   ├── 01_prediction.py
│   ├── 02_attention.py
│   ├── 03_explanation.py
│   ├── 04_counterfactual.py
│   ├── 05_evidence_path.py
│   └── 06_storyline.py
│
├── pages/                    ← Streamlit 멀티페이지
│   ├── 1_Overview.py
│   ├── 2_Integration.py
│   ├── 3_Message_Passing.py
│   ├── 4_Prediction.py
│   ├── 5_Explanation.py
│   ├── 6_Counterfactual.py
│   └── 7_Conclusion.py
│
├── components/               ← 재사용 UI 컴포넌트
│   ├── graph.py              ← PyVis 그래프 빌더
│   ├── gauge.py              ← P(sue) 게이지 차트
│   ├── cards.py              ← 스토리 카드, 메트릭 카드
│   ├── legend.py             ← 노드/엣지 범례
│   └── timeline.py           ← 결론 타임라인
│
├── utils/
│   ├── loader.py             ← results/*.json 로더
│   ├── graph_utils.py        ← top-k 엣지 필터 등
│   └── plot_utils.py         ← bar_chart, delta_bar 등
│
└── results/                  ← compute 결과물 (JSON)
    ├── graph.json
    ├── prediction.json
    ├── attention.json
    ├── explanation.json
    ├── counterfactual.json
    ├── evidence_path.json
    ├── storyline.json
    └── metadata.json
```

---

## compute/ 파이프라인 단계별 설명

| 스텝 | 파일 | 출력 | 내용 |
|------|------|------|------|
| 0  | `00_graph.py`         | `graph.json`         | k-hop 서브그래프, BFS 홉 거리, 레이아웃 |
| 10 | `00b_full_graph.py`   | `full_graph.json`    | 전체 그래프 (full context 시각화용) |
| 1  | `01_prediction.py`    | `prediction.json`    | 링크 예측 확률, Top-K 후보, 신뢰도 |
| 2  | `02_attention.py`     | `attention.json`     | 레이어별·관계별 GAT 어텐션 추출 |
| 3  | `03_explanation.py`   | `explanation.json`   | GNNExplainer + PGExplainer + SubgraphX (greedy LOO) |
| 4  | `04_counterfactual.py`| `counterfactual.json`| 엣지 타입별 / Top-K 개별 엣지 제거 시 ΔP |
| 5  | `05_evidence_path.py` | `evidence_path.json` | Plaintiff → Defendant 최단 증거 경로 |
| 6  | `06_storyline.py`     | `storyline.json`     | 모든 결과 종합 → 서사 카드 자동 생성 |
| -  | `run_all.py` (metadata)| `metadata.json`    | 모델 메타 정보 |

실행 방법:
```bash
cd src/data_266/0713/
python compute/run_all.py --pair 0 --khop 2 --topk 10
streamlit run app.py
```

---

## pages/ 페이지별 내용

| 페이지 | 파일 | 핵심 질문 |
|--------|------|----------|
| 1. Overview       | `1_Overview.py`       | GNN이 학습한 헤테로 그래프 전체 구조 |
| 2. Integration    | `2_Integration.py`    | 새 특허가 그래프에 어떻게 삽입되는가 |
| 3. Message Passing| `3_Message_Passing.py`| 정보가 어떻게 전파되는가 (홉 슬라이더) |
| 4. Prediction     | `4_Prediction.py`     | 누가 소송 예측 대상인가, Top-K 랭킹 |
| 5. Explanation    | `5_Explanation.py`    | 왜 이렇게 예측됐는가 (4가지 XAI 방법) |
| 6. Counterfactual | `6_Counterfactual.py` | 어떤 엣지를 제거하면 예측이 바뀌는가 |
| 7. Conclusion     | `7_Conclusion.py`     | 최종 서사·타임라인·증거 경로 종합 |

---

## components/ 주요 컴포넌트

- **`graph.py`** — PyVis 기반 인터랙티브 헤테로그래프. 노드 타입별 모양(⭐ plaintiff, ◆ defendant, ○ patent, □ company, △ IPC), 엣지 색상·굵기 인코딩, 어텐션/importance 임계값 필터.
- **`gauge.py`** — Plotly 반원 게이지로 P(sue) 시각화, 신뢰도 바.
- **`cards.py`** — 스토리 카드(아이콘+제목+설명), 메트릭 카드, info_banner.
- **`timeline.py`** — 결론 페이지용 수평 타임라인.
- **`legend.py`** — 노드·엣지 전체 범례 (사이드바 공통).

---

## 모델 정보

- **모델**: `SimpleCatHeteroGAT` (기존 viz_app/backend/model.py 재사용)
- **Hidden dim**: 128 → 64, **Heads**: [8, 4, 4, 2], **Layers**: 4
- **노드 타입**: patent, company, ipc
- **엣지 타입**: 8종 (cites, cited_by, sued, sued_by, owned_by, owns, has_ipc, rev_has_ipc)
- **데이터**: USPTO Patent Litigation (data_266)

---

## 오늘 진행 현황

### 완료
- [x] 아키텍처 설계 (`Patent_Litigation_XAI_Architecture.md`, `Patent_Litigation_XAI_SDD.md`)
- [x] `config.py` — 다크테마 색상 팔레트, PyVis/Plotly 기본 설정
- [x] `compute/common.py` — 공통 컨텍스트 로더, save_json 유틸
- [x] `compute/run_all.py` — 전체 파이프라인 오케스트레이터
- [x] `compute/00_graph.py` — k-hop 서브그래프 + BFS 홉 거리
- [x] `compute/00b_full_graph.py` — 전체 그래프 빌드
- [x] `compute/01_prediction.py` — 링크 예측 + Top-K
- [x] `compute/02_attention.py` — 네이티브 GAT 어텐션 추출 (훅 불필요)
- [x] `compute/03_explanation.py` — GNNExplainer + PGExplainer + SubgraphX
- [x] `compute/04_counterfactual.py` — 엣지 제거 ΔP 분석
- [x] `compute/05_evidence_path.py` — 증거 경로 탐색
- [x] `compute/06_storyline.py` — 서사 자동 생성
- [x] `components/graph.py` — PyVis 빌더
- [x] `components/gauge.py`, `cards.py`, `legend.py`, `timeline.py`
- [x] `utils/loader.py`, `graph_utils.py`, `plot_utils.py`
- [x] `pages/1~7` — 7개 Streamlit 멀티페이지 전부 구현
- [x] `app.py` — 홈 + 사이드바 (게이지, 예측 요약, 내비게이션)
- [x] `xai_results.json` — compute 파이프라인 실행 완료 (결과 JSON 생성됨)

### 미완료 / 남은 것
- [ ] `results/` 폴더의 분리된 JSON 파일들 생성 (현재는 단일 `xai_results.json` 사용 중, 새 compute/ 파이프라인은 `results/*.json` 기준)
- [ ] `compute/run_all.py` 실제 실행 및 `results/*.json` 확인
- [ ] 앱 실행 후 각 페이지 UI 검증
- [ ] 스크린샷 기록 (`스크린샷 2026-07-13 152517.png` 이후 추가 변경분)

---

## 핵심 설계 결정

1. **Compute/Viewer 완전 분리** — compute는 JSON만 생성, viewer는 JSON만 읽음. 모델 없이도 대시보드 동작 가능.
2. **네이티브 GAT 어텐션** — `GATConv(return_attention_weights=True)` 사용, forward hook 불필요.
3. **멀티페이지 Streamlit** — `pages/` 폴더 구조로 URL별 페이지 분리, 사이드바는 `app.py`에서 공통 렌더.
4. **4가지 XAI 병렬 비교** — GNNExplainer, GAT Attention, PGExplainer, SubgraphX를 동일 그래프에 적용하여 Agreement 탭에서 Pearson/Spearman 비교.
