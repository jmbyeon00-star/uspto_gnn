# MANTF-master 코드 구조 설명

**논문**: Han Wu et al., "A Multi-aspect Neural Tensor Factorization Framework for Patent Litigation Prediction"

**목적**: 원고 기업(plaintiff), 피고 기업(defendant), 특허(patent) 세 요소를 입력받아 특허 소송 발생 여부를 예측하는 모델.

---

## 전체 구조 개요

```
MANTF-master/
├── train.py                   # 진입점 (entry point)
├── solver.py                  # 학습/평가 루프 관리
├── utils.py                   # 루트 레벨 유틸리티 함수
└── module/
    ├── __init__.py            # 모듈 패키지 초기화
    ├── config.py              # 하이퍼파라미터 및 설정 관리
    ├── data.py                # 데이터셋 클래스 및 배치 구성
    ├── models.py              # MANTF 메인 모델 정의
    ├── layers.py              # GCN 레이어 정의
    ├── utils.py               # 모듈용 유틸리티 함수
    ├── evaluation_functions.py # 평가 지표 계산
    └── time_track.py          # 실행 시간 측정 데코레이터
```

---

## 파일별 상세 설명

### `train.py` — 학습 진입점

프로그램의 시작점. 설정을 불러오고, 데이터셋과 DataLoader를 구성한 뒤 `Solver`를 실행한다.

- `get_config()`로 CLI 인자를 파싱해 설정 객체 생성
- `MANTFDataset`으로 train/val/test 데이터셋 인스턴스 생성
- `mode`에 따라 동작 분기:
  - `train`: 학습 후 검증
  - `test`: 저장된 체크포인트로 테스트만 수행
  - `debug`: 학습 루프 디버깅
  - `optuna`: Optuna를 이용한 하이퍼파라미터 자동 탐색

---

### `solver.py` — 학습/평가 오케스트레이터

`Solver` 클래스가 모델 빌드, 학습 루프, 평가, 체크포인트 저장/불러오기를 담당한다.

**주요 메서드:**

| 메서드 | 역할 |
|--------|------|
| `build()` | 모델 생성, 특허 BERT 임베딩 로드, 옵티마이저 설정 |
| `train()` | BPR loss 기반 학습 루프 실행. 매 에폭마다 val 평가 수행 |
| `eval()` | val/test 데이터에 대해 AUC, Precision, Recall, F1, NDCG, MRR, HR@K 계산 |
| `save_model()` / `load_model()` | `.pkl` 형태로 체크포인트 저장/불러오기 |
| `optuna_train()` | Optuna trial 단위로 학습을 반복하는 하이퍼파라미터 탐색용 루프 |
| `encode_comp()` / `encode_pat()` | 기업/특허를 인코딩해 임베딩 벡터 반환 (추론용) |

**데이터 흐름 (학습 1 배치):**
1. 긍정 샘플 `(p_pla, p_def, p_pat)`과 부정 샘플 `(n_pla, n_def, n_pat)` 로드
2. 특허 BERT 임베딩 룩업 (title, abstract, claim, meta)
3. `MANTF.forward()` 호출 → 점수 및 손실 반환
4. 역전파 + 그래디언트 클리핑 + 옵티마이저 업데이트

**부정 샘플 유형 (neg_type):**
- `0`: 원고(plaintiff)를 다른 기업으로 교체
- `1`: 피고(defendant)를 다른 기업으로 교체
- `2`: 특허를 다른 특허로 교체

---

### `module/models.py` — MANTF 메인 모델

`MANTF` 클래스가 전체 모델을 정의한다.

**구성 요소:**

| 컴포넌트 | 설명 |
|----------|------|
| `_comp_g` | DGL 그래프 (기업 간 소송 네트워크) |
| `_comp_feat` | 기업 초기 특징 벡터 (freeze 가능) |
| `_gcn` | GCN_relu — 그래프 위에서 기업 임베딩을 학습 |
| `_comp_mask_pla` / `_comp_mask_def` | 원고/피고 역할별 임베딩 마스크 (sigmoid로 정규화) |
| `_fc_title`, `_fc_abstract`, `_fc_claim` | 특허 각 섹션의 BERT 임베딩을 변환하는 Linear 레이어 |
| `_fc_text` | title + abstract + claim을 결합한 텍스트 표현 생성 |
| `_mlp` | 텍스트 표현 + meta 임베딩 → 특허 임베딩 (factor space) |

**점수 계산 (Tensor Factorization):**
```
score = <pla, def> + <pla, pat> + <pat, def>
```
- 원고-피고, 원고-특허, 특허-피고 쌍별 내적의 합
- 각 임베딩에 가우시안 노이즈 추가 (Bayesian 확률적 모델링)

**손실 함수:**
- BPR (Bayesian Personalized Ranking): `loss = -log_sigmoid(pos_score - neg_score)`
- 정규화 항 5개 (loss1~loss5): 임베딩 norm 패널티, 레이어 weight norm 패널티

**분산 파라미터 (r_var, u_var, v_var, p_var, g_var, q_var):**
- 각 요소(소송결과, 원고, 피고, 특허, GCN, ...)의 불확실성을 제어
- lambda = 0.5 * (r_var / x_var) 형태로 정규화 강도를 자동 결정

---

### `module/layers.py` — GCN 레이어

`GCN_relu` 클래스. DGL의 `GraphConv`를 이용한 다층 GCN.

```
입력 특징 → GraphConv → ReLU → Dropout → [반복 N회] → 기업 임베딩
```

- 첫 번째 레이어: `in_feat_size → out_feat_size`
- 추가 레이어: `out_feat_size → out_feat_size` (gcn_layers - 1개)
- 기업 그래프의 이웃 기업 정보를 집계해 소송 관계 구조를 임베딩에 반영

---

### `module/config.py` — 설정 및 하이퍼파라미터 관리

`get_config()` 함수가 argparse로 CLI 인자를 파싱하고 `Config` 객체를 반환한다.

**주요 설정 그룹:**

| 그룹 | 파라미터 예시 |
|------|--------------|
| 학습 기본 | `batch_size`, `n_epoch`, `learning_rate`, `optimizer`, `clip` |
| 모델 구조 | `factor_size`, `dropout`, `gcn_layers`, `gcn_dropout`, `bert_size` |
| 분산/정규화 | `r_var`, `u_var`, `v_var`, `p_var`, `g_var`, `q_var`, `lambda0` |
| 데이터 경로 | `patent_idx_path`, `company_idx_path`, `comp_graph_path`, `comp_feat_path` |
| 특허 임베딩 | `pat_bert_title`, `pat_bert_abstract`, `pat_bert_claim`, `meta_embed_path` |
| 실험 제어 | `early_stop`, `random_seed`, `neg_types`, `description` |

실행 모드(`train`, `test`, `debug`, `optuna`)에 따라 저장 경로(`save_path`, `logdir`)를 자동 생성한다.

---

### `module/data.py` — 데이터셋 클래스

`MANTFDataset` (PyTorch `Dataset` 상속):
- CSV 파일에서 긍정/부정 샘플 쌍을 로드
- `neg_type`에 따라 원고/피고/특허 중 하나를 부정 샘플로 교체
- 학습 시 neg_type 0, 1의 샘플을 50% 확률로 드롭 → 세 태스크 비율 조절 (0:1:2 = 1:1:2)

`collate()`:
- Dataset의 샘플들을 배치 텐서로 변환하는 함수. `DataLoader`의 `collate_fn`으로 전달된다.

---

### `module/evaluation_functions.py` — 평가 지표

`leave_one_auc_scores_at_k()`:

- **Leave-one-out 방식**: 동일 (기업 쌍)에 속하는 긍정/부정 샘플을 그룹핑
- **계산 지표**: AUC, Precision@K, Recall@K, F1@K, NDCG@K, MRR@K, HitRate@K
- `neg_type`별로 독립적으로 계산 (원고 예측 / 피고 예측 / 특허 예측 성능을 분리 평가)
- K 기본값: `[1, 2, 3, 4, 5, 10, 20, 30, 40, 50]`

---

### `module/utils.py` — 유틸리티 함수 (모듈 내부용)

| 함수 | 역할 |
|------|------|
| `seed_torch(seed)` | 재현성을 위한 모든 랜덤 시드 고정 (random, numpy, torch, cuda) |
| `load_embedding(path, t)` | `.npy` 임베딩 파일 로드 (numpy 배열 반환) |
| `get_tokens_mapping(path, t)` | 기업/특허 ID ↔ 인덱스 매핑 딕셔너리 생성 |
| `padding_patent_text()` | 특허 텍스트를 고정 길이(300 토큰, 30 claims)로 패딩 |
| `timing` 데코레이터 | 함수 실행 시간을 ms 단위로 출력 |

---

### `module/time_track.py` — 시간 측정 데코레이터

`time_desc_decorator(desc)`:
- 함수에 붙이면 실행 시작 시 설명 문구를 출력하고, 종료 시 소요 시간을 출력
- `Solver.build()`, `Solver.train()` 등에 사용됨

---

### `utils.py` (루트) — 루트 레벨 유틸리티

`module/utils.py`와 거의 동일한 내용. 루트에서 직접 임포트해야 하는 경우를 위한 복사본.  
`load_embedding()`이 `dict`/`np` 두 가지 타입을 모두 지원하는 점이 약간 다르다.

---

## 방법론 설명: MANTF가 사용하는 핵심 기법

### 1. GCN (Graph Convolutional Network)

기업 노드들로 구성된 소송 네트워크 그래프 위에서 GCN을 적용해 기업 임베딩을 학습한다.

**핵심 아이디어**: 어떤 기업이 어떤 기업들과 소송 관계에 있는지(이웃 구조)가 해당 기업의 소송 성향을 설명한다는 가정 하에, GCN이 이웃 기업의 특징을 집계(aggregate)해 자신의 표현을 풍부하게 만든다.

```
h_v = ReLU(W · MEAN({h_u : u ∈ N(v) ∪ {v}}))
```

- 그래프 구성: 소송 비율(percentage-wise) 기반으로 연결된 기업 간 엣지 구성
- 양방향(bidirected) + self-loop 추가로 자신의 특징도 보존
- 원고/피고 역할이 다르므로 같은 GCN 출력에 역할별 마스크(sigmoid)를 곱해 역할을 분리

### 2. Tensor Factorization (텐서 분해)

원고(U) × 피고(V) × 특허(P) 의 3차원 텐서를 저차원으로 분해해 소송 발생 점수를 계산한다.

```
score(u, v, p) = <u, v> + <u, p> + <p, v>
```

이는 RESCAL/Tucker 분해의 단순화 형태로, 세 요소 간의 쌍별(pairwise) 상호작용을 모두 고려한다.

### 3. Bayesian Probabilistic Modeling (베이지안 확률적 모델링)

각 임베딩에 가우시안 노이즈를 추가해 불확실성을 모델링한다.

```
ũ = u + ε_u,  ε_u ~ N(0, σ_u²·I)
```

이는 MAP(Maximum A Posteriori) 추정에 해당하며, 노이즈 분산의 비율(`r_var / u_var`)이 L2 정규화 강도를 결정한다. 데이터가 적은 기업/특허에 대한 과적합을 방지하는 역할을 한다.

### 4. BPR Loss (Bayesian Personalized Ranking)

추천 시스템에서 널리 쓰이는 쌍별 랭킹 손실.

```
L = -log(σ(score_pos - score_neg))
```

"실제 소송 발생 (긍정 샘플)의 점수가 가짜 소송 (부정 샘플)의 점수보다 높아야 한다"는 직관을 학습한다.

### 5. Multi-aspect Patent Encoding (다중 관점 특허 인코딩)

특허를 세 가지 관점에서 BERT로 인코딩하고 통합한다:
- **Title**: 특허 제목 (핵심 기술 키워드)
- **Abstract**: 초록 (기술 요약)
- **Claims**: 청구항 (법적 권리 범위, 소송에서 가장 중요)
- **Meta (DeepWalk)**: 특허 인용 네트워크에서 학습된 구조적 임베딩

이 네 가지를 MLP로 결합해 단일 특허 임베딩을 생성한다.

### 전체 모델 흐름 요약

```
기업 초기 특징 (comp_feat)
        ↓
    GCN 레이어
        ↓
기업 임베딩 × 역할 마스크 → 원고 임베딩 / 피고 임베딩
                                        ↓
특허 BERT (title/abstract/claim) ─── MLP ──→ 특허 임베딩
특허 Meta (DeepWalk)          ─────────────↗
                                        ↓
        score = <pla, def> + <pla, pat> + <pat, def>
                                        ↓
                        BPR Loss (긍정 점수 > 부정 점수)
```
