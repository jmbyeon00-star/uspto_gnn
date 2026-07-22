# 266건 특허 소송 분석 — 작업 요약

---

## 1. 데이터 개요

### 원본 그래프 (`graph_266.graphml`)

| 구분 | 항목 | 수량 |
|------|------|-----:|
| **노드** | patent | 5,769 |
| | company | 1,061 |
| | case | 269 |
| | ipc | 218 |
| | **전체** | **7,317** |
| **엣지** | cites (특허 인용) | 9,545 |
| | owns (회사→특허 소유) | 5,789 |
| | co_litigation (같은 편 특허간) | 4,505 |
| | asserted_in (특허→소송 참여) | 1,460 |
| | has_ipc (특허→IPC 분류) | 1,545 |
| | sued (원고→피고 관계) | 256 |
| | **전체** | **23,100** |

### GNN 학습용 데이터셋

269개 케이스에서 원고특허 × 피고특허 크로스쌍을 모두 추출.

| 항목 | 수치 |
|------|-----:|
| 고유 특허 수 | 1,331 |
| 고유 회사 수 | 1,062 |
| IPC 코드 수 | 218 |
| IPC 엣지 | 1,545 |
| Citation 엣지 (소송특허 간) | 803 |
| 소유권 엣지 | 1,370 |
| **총 Sued 쌍 (P×D 크로스)** | **2,508** |

#### 케이스 수준 Train/Val/Test 분할 (80/10/10)

| 분할 | 케이스 수 | Sued 쌍 수 |
|------|----------:|----------:|
| Train | 215 | 2,198 |
| Val | 27 | 145 |
| Test | 27 | 165 |

> **케이스 수준 분할 이유**: 같은 케이스의 특허들이 train/test에 동시에 등장하면 데이터 누수 발생

---

## 2. 주요 관계 특성

### co_litigation vs sued 엣지 구분

| 엣지 | 의미 | 사용 여부 |
|------|------|----------|
| `co_litigation` | **같은 편** 특허 간 (P↔P, D↔D) | GNN 구조 엣지로 불사용 |
| `sued` | **원고→피고** 특허 간 (P→D) | 예측 타겟 |

co_litigation은 "같은 소송에서 같은 편"이라는 의미이므로 예측 타겟(sued 관계) 정보에 해당 → 학습 시 제외.

### Sued 쌍의 구조

케이스당 sued 쌍 수 = 원고특허 수 × 피고특허 수 (크로스 프로덕트)

- 케이스 66294: 39쌍 (가장 많음)
- 케이스 52634: 40쌍
- 케이스 14847: 16쌍
- 대부분의 케이스: 1~4쌍

### 텍스트 유사도 (FAISS BGE-M3 기준)

원고-피고 특허쌍의 cosine similarity:
- 소송쌍 (positive): mean = 0.667
- 랜덤쌍 (negative): mean = 0.510
- **Gap = 0.157**

→ 소송 관계의 특허들은 기술적으로 유사하다는 것을 정량적으로 확인.

---

## 3. 모델 설계

### 핵심 설계 원칙

| 결정 사항 | 이유 |
|----------|------|
| Case 노드 제외 | case 멤버십 자체가 예측 대상 정보 |
| co_litigation 엣지 제외 | 같은 편 여부도 예측 대상 정보 |
| P×D 크로스쌍 = 독립 샘플 | "케이스 단위 예측"이 아닌 "쌍 단위 예측" |
| IPC hard negative | 기술적으로 유사하지만 소송 아닌 쌍으로 난이도 향상 |

### 그래프 구조 (학습용)

```
노드:
  patent  (1,331개) — FAISS BGE-M3 1024d 임베딩 (고정)
  company (1,062개) — Learnable embedding
  ipc     (  218개) — Learnable embedding

엣지 (message passing용):
  patent  → patent   : cites / cited_by   (803개)
  patent  → ipc      : has_ipc            (1,545개)
  ipc     → patent   : rev_has_ipc        (1,545개)
  company → patent   : owns               (1,370개)
  patent  → company  : owned_by           (1,370개)

예측 타겟:
  patent  → patent   : sued (P→D)         (2,508쌍)
```

### 모델 아키텍처

**HeteroSAGE / HeteroGAT** (PyTorch Geometric)

```
Input:
  patent:  Linear(1024 → 256) + LayerNorm + ReLU
  company: Embedding(1062, 256)
  ipc:     Embedding(218, 256)

Conv layers (×3):
  [SAGEConv 또는 GATConv(heads=8)] per edge type
  → Residual + LayerNorm + ReLU + Dropout(0.2)

Output:
  Linear(256 → 128)
  → L2 normalize → cosine similarity로 소송 확률 계산
```

**SAGEConv vs GATConv 차이**

| | SAGEConv | GATConv |
|---|---|---|
| 이웃 집계 | 모든 이웃 평균 | α(patent_i, neighbor_j) 가중 합산 |
| IPC 가중치 | 모든 IPC 동일 | IPC별 다른 attention 학습 |
| 파라미터 수 | ~900K | ~1.8M |

### Loss 함수

| Loss | 학습 방식 |
|------|----------|
| **Margin loss** | pos vs IPC-hard neg(50%) + random neg(50%), k=4 |
| **InfoNCE** | 원고특허 기준 전체 1,331개 중 피고특허가 1위 되도록 직접 최적화, τ=0.07 |

---

## 4. 실험 결과

### 4-1. AUROC 기준 (pos vs 랜덤 1:1 비교)

| 모델 | Val AUROC (최고) | Test AUROC |
|------|----------------:|-----------:|
| FAISS baseline | — | 0.9301 |
| HeteroGAT + Margin | 0.8599 (epoch 20) | 0.9043 |

> HeteroGAT는 epoch 20 이후 val AUROC가 0.68까지 하락 → overfitting

### 4-2. Full Ranking 평가 (원고특허 기준 전체 1,331개 랭킹)

| 모델 | MRR | Hits@5 | Hits@10 | Hits@50 | Median rank |
|------|----:|-------:|--------:|--------:|-----------:|
| FAISS baseline | 0.182 | 0.333 | **0.515** | 0.673 | **10** |
| HeteroGAT + Margin | 0.104 | 0.182 | 0.327 | 0.606 | 31 |

> **FAISS가 Ranking에서 GNN을 크게 앞섬** — GNN이 ranking 기준으로 학습되지 않았기 때문

**FAISS Median rank = 10 / 1,331의 의미:**
- 원고특허와 텍스트가 가장 유사한 상위 10개 중에 실제 피고특허가 포함될 확률 51.5%
- 소송 관계의 특허들은 기술 내용이 유사하다는 것을 실증

### 4-3. 케이스별 AUROC (Test, 27 케이스)

| case_row_id | n_pairs | AUROC | pos_mean | gap |
|------------|--------:|------:|---------:|----:|
| 36495 | 2 | 0.500 | 0.247 | 0.037 |
| 55208 | 2 | 0.500 | 0.280 | 0.171 |
| 14984 | 4 | 0.750 | 0.653 | 0.405 |
| 66294 | 39 | 0.792 | 0.375 | 0.296 |
| 20238 | 4 | 0.813 | 0.592 | 0.558 |
| ... | ... | ... | ... | ... |
| (대부분) | 1~4 | 1.000 | — | — |

- Mean AUROC: **0.923**
- AUROC = 0.5 케이스(2건): n_pairs=2 → 통계적 노이즈 (샘플 부족)
- 66294 케이스(39쌍): 실제 어려운 케이스, 다양한 원고-피고 조합

---

## 5. 진행 중 실험

### 4가지 조합 비교 (Cell 12)

| 실험 | 모델 | Loss |
|------|------|------|
| SAGE + Margin  | HeteroSAGE | Margin (IPC-hard neg) |
| SAGE + InfoNCE | HeteroSAGE | InfoNCE (전체 랭킹 최적화) |
| GAT  + Margin  | HeteroGAT  | Margin (IPC-hard neg) |
| GAT  + InfoNCE | HeteroGAT  | InfoNCE (전체 랭킹 최적화) |

평가 지표: AUROC, AUPRC, MRR, Hits@5/10/50, Median rank

---

## 6. 다음 단계 (예정)

1. **4실험 결과 비교** → 최적 모델 선택
2. **Attention weight 분석** (GAT가 최적이면)
   - 엣지 타입별 (citation / IPC / ownership) 평균 attention
   - 케이스별 "가장 중요한 IPC 코드 / citation" 도출
3. **GNNExplainer** → 특정 소송쌍 예측 근거 설명
4. **Graphistry 시각화** → 중요도 점수를 노드/엣지 색으로 표현

---

## 7. 파일 구조

```
src/data_266/
  data_266.ipynb       — 데이터 탐색 + Graphistry 시각화
  gnn_266.ipynb        — GNN v1 (설계 오류 포함, 참고용)
  gnn_266_v2.ipynb     — GNN v2 (현재 메인)
src/
  graph_266.graphml    — 전체 이종 그래프
data/samples/
  both_side_patents.csv
  patent_ownership_matched_4.csv
```
