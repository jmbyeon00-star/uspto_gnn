# 0922PSEUDO_ITERATIVE_MLP 전체 정리

> 기준일: 2026-09-30  
> 대상 폴더: `0922PSEUDO_ITERATIVE_MLP/`  
> 제외 폴더: `negative_mining/`, `iterative_distmult/`

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [폴더 구조](#2-폴더-구조)
3. [데이터 구성](#3-데이터-구성)
   - 3.1 원시 데이터 및 외부 파일
   - 3.2 Snapshot
   - 3.3 노드별 입력 피처
   - 3.4 그래프 엣지 타입 (18종)
   - 3.5 C3 Fixed Split (SUP / VAL)
   - 3.6 Unified Hard-Negative Pool
   - 3.7 Test GT
4. [모델 구조](#4-모델-구조)
   - 4.1 Input Projection
   - 4.2 StreamingHeteroLayer (SAGEConv 기반)
   - 4.3 Residual + LayerNorm
   - 4.4 MLPDecoder
   - 4.5 파라미터 요약
5. [학습 방법 (Training)](#5-학습-방법-training)
   - 5.1 시간 분할 Fold 구조
   - 5.2 학습 데이터 구성 (SUP ALL)
   - 5.3 Negative Sampling 전략
   - 5.4 Loss 및 옵티마이저
   - 5.5 Warm-start (Fold Inheritance)
6. [Iterative Pseudo-label 파이프라인](#6-iterative-pseudo-label-파이프라인)
   - 6.1 개념
   - 6.2 1단계: score_case12.py — Case1/2 스코어링
   - 6.3 2단계: analyze_top1.py — Top1 분석
   - 6.4 3단계: select_pseudo.py — Pseudo 선택
   - 6.5 4단계: build_round_data.py — 다음 Round 데이터 빌드
   - 6.6 5단계: train_round.py — 다음 Round 학습
   - 6.7 Pseudo MP vs Pseudo SUP 구분
   - 6.8 누적 방식 및 Leakage 방지
7. [Predecessor (Fold 간 계승)](#7-predecessor-fold-간-계승)
   - 7.1 개념
   - 7.2 build_inherited.py — Pseudo 계승 준비
   - 7.3 train_predecessor.py — 계승 후보 학습 및 선택
   - 7.4 selected.pkl — 최적 후보 확정
8. [Validation 방법](#8-validation-방법)
   - 8.1 Fixed VAL Set 구성
   - 8.2 평가 지표
   - 8.3 Early Stopping
9. [Test 방법 (Filtered Full Ranking)](#9-test-방법-filtered-full-ranking)
   - 9.1 Test GT 구성
   - 9.2 Filtered Full Ranking 절차
   - 9.3 출력 지표
10. [전체 파이프라인 실행 순서](#10-전체-파이프라인-실행-순서)
11. [하이퍼파라미터 요약](#11-하이퍼파라미터-요약)
12. [주요 파일 역할 요약](#12-주요-파일-역할-요약)

---

## 1. 프로젝트 개요

이 프로젝트는 미국 특허 소송(USPTO litigation)에서 **원고 특허 → 피고 특허** 간의 소송 관계를 예측하는 링크 예측 모델이다.

핵심 아이디어:
- **GNN(SAGEConv)** 기반 이종 그래프(heterogeneous graph)로 특허·IPC·기업·소송케이스 노드를 인코딩
- **Iterative Pseudo-label training**: 모델이 아직 레이블 없는 Case1/2 케이스(한 쪽 특허만 알려진 케이스)를 스스로 스코어링하여 고신뢰(score ≥ 0.99) pair를 pseudo-positive로 추가, 다시 학습 — 이를 10 round 반복
- **Temporal Fold 구조**: 시간 순서대로 학습(cutoff 2004/2006/2008/2010 → test 2012~2014), 각 fold를 독립적으로 iterative 학습 후 다음 fold에 최고 체크포인트를 warm-start로 넘김 (Predecessor 선택)
- **Filtered Full Ranking** 평가: test 시 모든 특허를 후보로 놓고 MRR, Hits@K 계산

---

## 2. 폴더 구조

```
0922PSEUDO_ITERATIVE_MLP/
├── config.py                          # 전역 하이퍼파라미터, 경로
├── data.py                            # 피처 빌드, 엣지 정의, 스냅샷 로드
├── model.py                           # GNN 모델 (StreamingHeteroLayer + MLPDecoder)
├── run_2008_to_2010.py                # 2008 R0~R10 → 2010 Predecessor 비교 실행 스크립트
├── test_full_ranking.py               # Final Test (2012 snapshot, 2013-2014 GT)
│
├── fixed_data/                        # 사전 계산된 고정 데이터 (학습 중 변하지 않음)
│   ├── c3_pools/
│   │   └── cutoff_{2004,2006,2008,2010}/
│   │       └── c3_data.pkl            # C3 케이스 SUP/VAL 7:3 fixed split + 음수 pool
│   ├── unified_pools/
│   │   ├── pool_{cutoff}.pkl          # Fold별 unified hard-negative pool
│   │   └── detail_{cutoff}.csv        # Pool 빌드 상세 통계
│   ├── val_sets/
│   │   └── val_{cutoff}.pkl           # 사전 빌드된 Val record (hard/random neg 포함)
│   ├── case2_candidates/
│   │   └── cases_{cutoff}.pkl         # Case2 후보 src 정보
│   └── test/
│       └── test_gt_2013_2014.pkl      # Final Test GT (2013-2014 소송 pair)
│
├── iterative/                         # Iterative pseudo-label 학습 모듈
│   ├── build_round_data.py            # Round별 학습 데이터 조립
│   ├── train_round.py                 # Round별 모델 학습
│   ├── score_case12.py                # 현재 모델로 Case1/2 전체 스코어링
│   ├── analyze_top1.py                # Top1 점수 분포 분석 및 parquet 저장
│   ├── select_pseudo.py               # 임계값 기반 pseudo 선택 및 검증
│   ├── build_inherited.py             # Source fold pseudo → target fold remap
│   ├── train_predecessor.py           # Predecessor 후보(R0~R10) 학습 및 비교
│   ├── build_unified_pools.py         # Unified hard-negative pool 최초 빌드
│   ├── pool_manager.py                # UnifiedPoolManager (lazy-build 포함)
│   ├── state.py                       # FoldState (round 진행 중 상태 저장)
│   ├── fold_inheritance_audit.py      # Fold 계승 감사 도구
│   ├── analyze_scores.py              # 스코어 분포 분석 도구
│   └── build_test_inherited.py        # Test fold 상속 pseudo 빌드
│
├── baseline/                          # Round 0 (no pseudo) baseline 결과
│   └── cutoff_{2004,2006,2008,2010}/
│       ├── checkpoint.pt
│       ├── metrics.json
│       └── round_data_summary.json
│
├── runs/                              # 실제 학습 결과물 (Round별)
│   └── cutoff_{Y}/
│       └── round_{NN}/
│           ├── round_data.pkl         # Round 입력 데이터 (SUP/VAL/pseudo/neg pool)
│           ├── round_data_summary.json
│           ├── checkpoint.pt          # Best VAL MRR 체크포인트
│           ├── metrics.json           # 학습 결과 지표
│           ├── scored_candidates.parquet   # Case1/2 전체 스코어링 결과
│           ├── top1_candidates.parquet     # Top1 후보 요약
│           ├── top1_threshold_counts.csv   # 임계값별 선택 수
│           └── pseudo_selected_for_round_{N+1}.pkl
│
├── checkpoint_only_test/              # Checkpoint 단독 평가 스크립트 모음
├── analyze_ipc3_fallback_sequence.py  # IPC3 fallback 분석 도구
├── analyze_ipc3_hard_pool.py          # Hard pool IPC3 구성 분석 도구
├── expand_unified_pools.py            # Unified pool 확장 스크립트
└── 검증.ipynb                          # 수동 검증 노트북
```

---

## 3. 데이터 구성

### 3.1 원시 데이터 및 외부 파일

| 파일 | 내용 |
|------|------|
| `data_gnn/bge_pca128_feat22_concat_5m.npy` | 약 500만 특허의 임베딩 행렬 (BGE-PCA128 + feat22 concat) |
| `data_gnn/bge_pca128_feat22_concat_5m_pids.npy` | 위 임베딩의 특허 ID 배열 |
| `data_gnn/ipc_subclass_ids.npy` | IPC subclass 코드 배열 |
| `data_gnn/ipc_node_x.npy` | IPC 노드 피처 행렬 (L2 정규화 적용) |
| `data/with_counter/gnn_usable_cases_v2.parquet` | 사용 가능한 소송 케이스 메타데이터 (date_filed, case_type, assignee 등) |
| `data/with_counter/gnn_case_patents_v2.parquet` | 케이스별 특허 목록 (role: plaintiff/defendant) |
| `data/with_counter/ownership_chain_v2.parquet` | assignee별 특허 소유 이력 (start_dt, end_dt) |
| `0916PSEUDO_ONESHOT/snapshots_clean_0917/snapshot_{year}*.pt` | 연도별 그래프 스냅샷 (PyG HeteroData) |

### 3.2 Snapshot

`snapshot_{year}.pt` (또는 `snapshot_{year}_v2.pt`)는 해당 연도까지의 누적 이종 그래프다.

```
snapshot["data"]          : PyG HeteroData  (모든 노드/엣지)
snapshot["meta"]          : {"snapshot_year": int}
snapshot["pat_list"]      : 특허 노드 순서 (로컬 인덱스 매핑 기준)
snapshot["ipc_list"]      : IPC 노드 순서
snapshot["pat2idx"]       : {patent_id: local_idx}  (dict)
snapshot["sue_meta"]      : DataFrame — 소송 메타데이터 (patent_plt, patent_def, lawsuit_year 등)
```

- 각 Fold의 cutoff 연도(`2004`, `2006`, `2008`, `2010`, `2012`)마다 별도 스냅샷 로드
- `build_features()` 함수가 스냅샷을 받아 노드별 텐서 피처 딕셔너리를 반환

### 3.3 노드별 입력 피처

`build_features(snapshot, static)` → `feats: Dict[str, Tensor]`

| 노드 타입 | 피처 차원 | 구성 방법 |
|-----------|-----------|-----------|
| `patent` | 150차원 (BGE-PCA128 + feat22) | 전역 임베딩에서 로컬 인덱스로 조회 → Z-score 정규화(μ=0, σ=1, clip ±3) → L2 정규화 |
| `ipc` | (ipc_node_x 차원) | 전역 IPC 피처에서 조회 → L2 정규화 |
| `company` | (snapshot["data"]["company"].x 차원) | 스냅샷 내 company 노드 피처 그대로 사용 |
| `case` | 1차원 | 소송 제기 연도 기준 나이 `(cutoff - year) / 20.0` (0~1 스케일, clip 0~20년) |

특허 피처 히트율: 약 96.4% (cutoff 2008 기준 `767,795 / 796,504`)

### 3.4 그래프 엣지 타입 (18종)

`data.py`의 `EDGE_TYPES` 리스트:

| # | 엣지 타입 (src, rel, dst) | 설명 |
|---|--------------------------|------|
| 1 | `(patent, cites, patent)` | 특허 인용 |
| 2 | `(patent, cited_by, patent)` | 특허 피인용 |
| 3 | `(patent, sue, patent)` | 소송 (원고특허 → 피고특허) ← **예측 타겟** |
| 4 | `(patent, sued_by, patent)` | sue 역방향 |
| 5 | `(patent, has_ipc, ipc)` | IPC 분류 |
| 6 | `(ipc, rev_has_ipc, patent)` | has_ipc 역방향 |
| 7 | `(patent, owns, company)` | 특허 소유 기업 |
| 8 | `(company, rev_owns, patent)` | owns 역방향 |
| 9 | `(patent, prev_owns, company)` | 이전 소유 기업 |
| 10 | `(company, rev_prev_owns, patent)` | prev_owns 역방향 |
| 11 | `(case, plaintiff, company)` | 소송케이스-원고기업 |
| 12 | `(company, rev_plaintiff, case)` | plaintiff 역방향 |
| 13 | `(case, defendant, company)` | 소송케이스-피고기업 |
| 14 | `(company, rev_defendant, case)` | defendant 역방향 |
| 15 | `(case, plaintiff_patent, patent)` | 소송케이스-원고특허 |
| 16 | `(patent, rev_plaintiff_patent, case)` | plaintiff_patent 역방향 |
| 17 | `(case, defendant_patent, patent)` | 소송케이스-피고특허 |
| 18 | `(patent, rev_defendant_patent, case)` | defendant_patent 역방향 |

#### 엣지 가시성 규칙 (build_visible_edge_dict)

학습 중 **정보 누수(leakage) 방지**를 위한 엣지 마스킹:

- `sue`/`sued_by`: cutoff 연도 **미만**(< cutoff_year)의 역사적 소송만 message-passing에 사용. cutoff 연도 소송(= 현재 fold C3)은 모두 숨김
- Case 관련 엣지(`case↔patent`, `case↔company`): **현재 fold의 숨겨진 케이스**(hidden_current_cases)와 연결된 엣지 제거
- 역사적 sue MP에서 Fixed VAL positive pair도 별도 마스킹 (`mask_fixed_val_from_historical_sue`)

### 3.5 C3 Fixed Split (SUP / VAL)

`fixed_data/c3_pools/cutoff_{Y}/c3_data.pkl`

C3 케이스(원고 특허와 피고 특허가 모두 알려진 케이스) 를 cutoff 연도 기준으로 **고정(fixed)**하여 한 번 7:3으로 분할:

| 분할 | 비율 | 역할 |
|------|------|------|
| `sup_pairs` (train_records) | ~70% | 학습 positive (SUP real) |
| `val_pairs` (val_records) | ~30% | 검증 positive (VAL fixed) — 모든 round에서 동일 |

추가 필드:
- `all_positive_pairs`: 스냅샷 내 알려진 모든 positive pair (negative sampling 차단용)
- `val_reserved_pairs`: VAL 관련 예약 pair (negative pool에서도 제외)
- `num_patents`: fold 내 특허 노드 수

예시(cutoff=2008): `sup_pairs=35, val_pairs=41`

### 3.6 Unified Hard-Negative Pool

`fixed_data/unified_pools/pool_{cutoff}.pkl`

각 positive src 특허에 대해 **최대 2,000개**의 hard-negative 후보를 사전 계산해 저장.

#### 후보 선발 우선순위 (build_unified_pools.py)

**Hard 그룹 (B > A 순으로 채움):**

| 단계 | 집합 | 정의 |
|------|------|------|
| B1 | `counterparty_portfolio ∩ (IPC3 ∪ citation1)` | 상대방 기업 포트폴리오 & 1-hop 인용 & IPC3급 동일 |
| B2 | `counterparty_portfolio ∩ (IPC3 ∪ citation≤2)` | B1 확장 (2-hop) |
| B3 | `counterparty_portfolio ∩ (IPC3 ∪ citation≤3)` | B2 확장 (3-hop) |
| A1 | `IPC3 ∩ citation1` | IPC subclass 동일 & 1-hop 인용 |
| A2 | `IPC3 ∩ citation≤2` | A1 확장 |
| A3 | `IPC3 ∩ citation≤3` | A2 확장 |

**Fallback (hard 2,000개 미달 시):**
`citation1 → citation≤2 → citation≤3 → IPC4 → IPC3` 순으로 보충

- `counterparty`: 소송 당시 상대방 기업이 사건 발생일(event_dt) 기준으로 실제 보유한 포트폴리오 (ownership_chain_v2 시간 필터 적용)
- `real_positive_by_src`: 알려진 실제 positive는 pool에서 사전 제거

### 3.7 Test GT

`fixed_data/test/test_gt_2013_2014.pkl`

- test 기간: **2013~2014년** 소송 데이터
- 2012 snapshot 기준으로 patent ID를 로컬 인덱스로 매핑
- `test_patent_pairs`: `{(str, str), ...}` — (원고특허, 피고특허) set

---

## 4. 모델 구조

`model.py` — `Model` 클래스

### 4.1 Input Projection

```python
self.proj = nn.ModuleDict({
    ntype: nn.Linear(dim, hidden)   # hidden = 128
    for ntype, dim in input_dims.items()
})
```
노드 타입별로 개별 Linear로 hidden(128) 차원으로 투영 후 ReLU 활성화.

### 4.2 StreamingHeteroLayer (SAGEConv 기반)

```python
class StreamingHeteroLayer(nn.Module):
    """
    관계별 SAGEConv를 순서대로 실행하고 동일 dst 타입 결과를 sum 집계.
    torch.stack 대신 streaming sum을 사용하여 CUDA OOM 방지.
    """
    convs = nn.ModuleDict({
        "__".join(etype): SAGEConv((hidden, hidden), hidden)
        for etype in edge_types
    })
```

- 18개의 엣지 타입 각각에 독립적인 `SAGEConv((128,128), 128)` 보유
- 같은 dst 타입으로 들어오는 여러 관계의 출력을 **streaming sum**으로 누적
- 엣지가 없는 관계는 skip (0 벡터 대신 아무것도 더하지 않음)

### 4.3 Residual + LayerNorm (2-layer)

```
encode():
  h = { k: ReLU(proj[k](feat_k)) }       # 투영

  o1 = layer1(h, edge_dict)               # 1st SAGEConv
  h1 = { k: dropout(ReLU(norm1[k](o1[k] + h[k]))) }   # residual + LN + dropout

  o2 = layer2(h1, edge_dict)              # 2nd SAGEConv
  h2 = { k: norm2[k](o2[k] + h1[k]) }    # residual + LN (dropout 없음)

  return h2["patent"]                     # patent 노드 임베딩만 반환
```

- **2층 GNN** (2-hop 이웃까지 정보 수집)
- 각 layer마다 residual connection + LayerNorm
- 1층 후 Dropout(p=0.30), 2층 후 Dropout 없음
- 최종 출력은 `patent` 노드 임베딩만 (`z_patent`: shape `[N_pat, 128]`)

### 4.4 MLPDecoder

```python
class MLPDecoder(nn.Module):
    mlp = nn.Sequential(
        nn.Linear(128 * 2, 128),   # concat(z_src, z_dst)
        nn.ReLU(),
        nn.Linear(128, 1),         # scalar logit
    )
```

- `z_src`와 `z_dst`를 concat(256차원) → 128 → 1 (scalar logit)
- `model.score(z, src_idx, dst_idx)` → logit tensor (sigmoid 전 값)

### 4.5 파라미터 요약

| 구성 요소 | 레이어 수 | 비고 |
|-----------|-----------|------|
| Input Projection | 4개 (patent/ipc/company/case) | 노드 타입별 개별 Linear |
| StreamingHeteroLayer × 2 | 18 × 2 = 36개 SAGEConv | 층당 관계 수 × 2층 |
| LayerNorm × 2 | 4 × 2 = 8개 | 노드 타입별, 층별 |
| MLPDecoder | 2층 FC | 256→128→1 |
| hidden_dim | 128 | config.HIDDEN |
| dropout | 0.30 | config.DROPOUT |

---

## 5. 학습 방법 (Training)

### 5.1 시간 분할 Fold 구조

`config.TRAIN_FOLDS`, `config.TEST_FOLD`

| Fold | Snapshot (cutoff) | 학습 타겟 연도 | 비고 |
|------|-------------------|----------------|------|
| Fold 1 | 2004 | 2005~2006 | 최초 fold (predecessor 없음) |
| Fold 2 | 2006 | 2007~2008 | Fold 1 체크포인트 warm-start 후보 |
| Fold 3 | 2008 | 2009~2010 | Fold 2 체크포인트 warm-start 후보 |
| Fold 4 | 2010 | 2011~2012 | Fold 3 체크포인트 warm-start 후보 |
| Test | 2012 | 2013~2014 | Fold 4 체크포인트 warm-start 후보 |

각 fold는 독립적으로 **Round 0 ~ Round 10**까지 iterative pseudo-label 학습을 진행한다.

### 5.2 학습 데이터 구성 (SUP ALL)

`build_round_data.py` → `round_data.pkl`의 `sup_all`

```
SUP ALL = SUP real (C3 7:3 split의 training side)
        + SUP pseudo (이전 rounds에서 선택된 pseudo 중 target_start~target_end 연도 해당분)
```

- SUP real과 SUP pseudo 간 중복 제거
- VAL fixed와 겹치는 pair는 모두 제거 (VAL 보호 최우선)
- real SUP ∩ VAL이 발견되면 실시간 제거 후 경고

### 5.3 Negative Sampling 전략

`sample_negs()` — 매 epoch, 매 positive pair마다 호출

```
per positive (s, d):
  hard_pool = unified_negative_pool[s]  (사전 계산된 hard 후보 최대 2,000개)
  
  blocked = {s} | all_positives[s] | val_reserved[s]
  
  hard_candidates = hard_pool - blocked  → shuffle → 앞 N_HARD_TRAIN(=2)개
  
  # hard quota 미달 시 전체 특허에서 랜덤 보충
  random_candidates = random_global(0~N_pat) - blocked - hard_candidates
                      최대 N_RANDOM_TRAIN(=3)개
  
  negatives = hard_candidates + random_candidates  # 총 5개
```

- **총 5개 negative** per positive (hard 2 + random 3)
- Round가 진행될수록 pseudo positive가 blocked에 추가되므로 hard pool이 점진적으로 줄어듦 (`active_negative_pools`)

### 5.4 Loss 및 옵티마이저

```python
# 모든 positive + negative를 concat하여 BCE loss
logits = torch.cat([pos_scores, neg_scores])
labels = torch.cat([ones, zeros])
loss = F.binary_cross_entropy_with_logits(logits, labels)

optimizer = AdamW(lr=1e-3, weight_decay=1e-4)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

| 항목 | 값 |
|------|-----|
| Loss | Binary Cross-Entropy with Logits |
| Optimizer | AdamW |
| LR | 1e-3 |
| Weight Decay | 1e-4 |
| Gradient Clipping | max_norm = 1.0 |
| Max Epochs | 1,000 |
| Val Check 주기 | 10 epochs |
| Early Stopping Patience | 60 (validation 미개선 횟수) |

### 5.5 Warm-start (Fold Inheritance)

- **Round 0**: 이전 fold에서 Predecessor 선택 과정을 통해 선정된 최적 체크포인트로 warm-start
  - `cutoff=2004` 최초 fold만 random init
  - 이후 fold는 반드시 이전 fold의 최적 R0~R10 중 하나를 warm-start로 사용
- **Round 1+**: 직전 round(`round_{N-1}/checkpoint.pt`) 자동 warm-start
- Warm-start 시 모델 weights를 그대로 로드하고 새 데이터(pseudo 포함)로 재학습

---

## 6. Iterative Pseudo-label 파이프라인

### 6.1 개념

```
Case 유형:
  Case3 (C3): 원고 특허 & 피고 특허 모두 알려진 케이스 → Ground Truth
  Case1 (C1): 원고 특허만 알려진 케이스 → 피고 특허 후보 예측
  Case2 (C2): 피고 특허만 알려진 케이스 → 원고 특허 후보 예측
```

Case1/2는 정답이 없지만, 현재 모델이 **가장 높은 score를 준 후보 특허**가 실제 소송 상대 특허일 가능성이 높다. 이 Top1 후보가 충분히 고확신(score ≥ 0.99)이면 **pseudo-positive**로 채택하여 다음 round 학습에 포함한다.

### 6.2 1단계: score_case12.py — Case1/2 스코어링

**입력:**
- 현재 round의 checkpoint (`runs/cutoff_{Y}/round_{R}/checkpoint.pt`)
- 소송 케이스 테이블 (`gnn_usable_cases_v2.parquet`)
- 케이스별 특허 목록 (`gnn_case_patents_v2.parquet`)
- 소유권 이력 (`ownership_chain_v2.parquet`)

**처리 절차:**
1. 현재 모델로 그래프 인코딩 → `z_patent` 임베딩
2. case_type이 case1/case2이고 date_filed ≤ target_end인 케이스 필터링
3. 각 케이스마다:
   - Case1: known_patent = 원고 특허, candidate = 상대방 기업(event_dt 기준 포트폴리오)
   - Case2: known_patent = 피고 특허, candidate = 상대방 기업 포트폴리오
4. 각 (known_patent, candidate_patent) 쌍에 대해 `model.score()` 계산
5. 케이스 내에서 score 기준 rank 부여

**출력:** `runs/cutoff_{Y}/round_{R}/scored_candidates.parquet`
- 모든 (케이스, known_patent, candidate_patent) 조합의 logit, score, rank

### 6.3 2단계: analyze_top1.py — Top1 분석

**처리:**
1. `scored_candidates.parquet` 로드
2. 각 (case_row_id, known_patent) query에서 rank=1인 행 추출 → Top1 후보
3. rank=2를 merge하여 `margin = top1_score - top2_score` 계산
4. threshold별(0.50~0.99) 후보 수 집계
5. bucket(mp/sup)별, case_type별 점수 분포 출력

**출력:**
- `top1_candidates.parquet`: 다음 단계(select_pseudo)에서 사용
- `top1_threshold_counts.csv`: 임계값별 통계

### 6.4 3단계: select_pseudo.py — Pseudo 선택

**임계값:** `THRESHOLD = 0.99` (sigmoid 확률 기준)

**처리 순서 (단계별 필터링):**

| 단계 | 제거 조건 |
|------|-----------|
| 1 | `top1_score < threshold` |
| 2 | src/dst/event_year 결측 |
| 3 | `src == dst` (자기 자신 pair) |
| 4 | temporal window 외부 (`event_year > cutoff` and `event_year < target_start` or `> target_end`) |
| 5 | 동일 (plt_patent, def_patent)에서 여러 케이스 → 최고 score만 유지 |
| 6 | `all_positive_pairs`와 충돌 (real positive 재선택 방지) |
| 7 | `val_reserved_pairs`와 충돌 |
| 8 | 이전 rounds/inherited pseudo와 특허 ID 기준 충돌 |

**출력:** `pseudo_selected_for_round_{R+1}.pkl`
```python
{
    "pseudo_mp": [records ...],   # event_year <= cutoff → MP pseudo
    "pseudo_sup": [records ...],  # target_start <= event_year <= target_end → SUP pseudo
}
```

### 6.5 4단계: build_round_data.py — 다음 Round 데이터 빌드

**처리:**
1. Fixed C3 SUP/VAL 로드 (`c3_data.pkl`)
2. Unified negative pool 로드
3. 이전 모든 rounds의 pseudo 누적 로드 (`load_previous_pseudo`: round 0~R-1 전부)
4. Inherited pseudo 로드 (`load_inherited_pseudo`: predecessor fold에서 계승)
5. 모든 pseudo를 `split_pseudo_by_time()`으로 MP/SUP 분리
6. `clean_pseudo()`: VAL 보호 → real SUP 중복 제거 → MP↔SUP 중복 제거
7. `SUP ALL = SUP real + SUP pseudo` (dedup)
8. `build_active_negative_pool()`: unified pool에서 모든 positive/reserved 제거
9. 6중 검증(leakage 확인) 후 저장

**검증 체크 항목:**
- real SUP ∩ VAL = 0
- pseudo_MP ∩ VAL = 0
- pseudo_SUP ∩ VAL = 0
- pseudo_MP ∩ pseudo_SUP = 0
- real_SUP ∩ pseudo_SUP = 0
- active_negative에 positive 없음
- active_negative에 val_reserved 없음

**출력:** `round_data.pkl`
```python
{
    "sup_all": [...],             # 학습 positive 전체
    "val_fixed": [...],           # 검증 positive (고정)
    "mp_pseudo": [...],           # pseudo MP (graph edge에 추가될 것들)
    "sup_pseudo": [...],          # pseudo SUP (학습에 추가될 것들)
    "negative_pools": {src: [...]}, # active negative pool
    "val_reserved_pairs": set,    # negative sampling 금지 pair
    ...
}
```

### 6.6 5단계: train_round.py — 다음 Round 학습

학습 전 엣지 처리:
1. `mask_fixed_val_from_historical_sue()`: 역사적 sue MP에서 VAL pair 제거
2. `add_mp()`: 역사적 sue + pseudo MP를 unique union하여 message-passing edge 구성

이후 표준 학습 루프 실행 → checkpoint 저장.

### 6.7 Pseudo MP vs Pseudo SUP 구분

```
date_filed(event_year) 기준:

  event_year <= cutoff  → pseudo MP
    목적: 그래프 엣지(message-passing)에 추가하여 표현 개선
    효과: 2층 GNN이 이 엣지를 통해 정보 전파

  target_start <= event_year <= target_end  → pseudo SUP
    목적: 학습 positive pair(SUP ALL)에 추가
    효과: 더 많은 positive 예시로 직접 학습
```

pseudo MP와 pseudo SUP는 동일 pair가 되지 않도록 중복 제거 (MP 우선).

### 6.8 누적 방식 및 Leakage 방지

- Pseudo는 **누적**: round 0에서 선택 → round 1 데이터에 포함 → round 2 데이터에도 포함 (모든 이전 round pseudo 합산)
- **Patent ID 기준** identity: cross-snapshot, cross-round에서 동일 pair 중복 방지
- VAL fixed는 어떤 round에도 건드리지 않음 (최우선 보호)
- Predecessor fold pseudo가 현재 fold GT와 겹치면 제거 (build_inherited.py에서 처리)

---

## 7. Predecessor (Fold 간 계승)

### 7.1 개념

Fold N에서 R0~R10까지 학습한 체크포인트 중 어느 것이 **다음 fold(N+2)**의 초기값으로 가장 좋은가?를 결정하는 과정.

- Source fold: cutoff Y (R0~R10 체크포인트 보유)
- Target fold: cutoff Y+2

**왜 R0이 아닌 Rx가 더 좋을 수도 있는가?**
- pseudo가 너무 많으면 noise가 많아 성능이 오히려 저하될 수 있음
- 최적 round는 target fold의 validation으로 직접 확인

### 7.2 build_inherited.py — Pseudo 계승 준비

**입력:**
- source fold의 `round_{R}/round_data.pkl` (R = 0~10)
- target fold snapshot (`snapshot_{Y+2}.pt`)
- target fold의 C3 fixed split

**처리:**
1. source pseudo 전체 수집 (`mp_inherited + mp_pseudo + sup_pseudo`)
2. patent ID 기준 dedup
3. target snapshot에 없는 특허(patent not alive) → missing으로 기록, 제외
4. target fold GT MP / GT SUP / VAL과 겹치는 pair → blocked로 분류, 제거
5. 남은 pair를 target snapshot 인덱스로 remap

**출력:** `runs/predecessor_{from}_to_{to}/source_round_{R}/inherited.pkl`
```python
{
    "inherited_records": [...],        # 원본 메타데이터 + target idx
    "inherited_pairs": [(src,dst),...], # target snapshot 인덱스 pair
    "inherited_patent_pairs": [("p1","p2"),...], # patent ID pair (영구 identity)
    "blocked_pairs": {...},             # 차단된 pair 감사용
    "stats": {...}
}
```

### 7.3 train_predecessor.py — 계승 후보 학습 및 비교

각 source round (R=0~10)에 대해:
1. source round 체크포인트를 **warm-start**로 로드
2. target fold snapshot + inherited pseudo MP를 그래프에 추가
3. target fold C3 SUP + target fold negative pool로 재학습
4. target fold **fixed VAL MRR**를 기준으로 best checkpoint 저장

```
실행 결과:
  runs/predecessor_{src}_to_{tgt}/
    source_round_00/  checkpoint.pt, metrics.json
    source_round_01/  ...
    ...
    source_round_10/
    comparison.csv    (모든 source round의 VAL MRR 비교)
    selected.pkl      (최고 VAL MRR source round 정보)
    selected_checkpoint.pt
```

### 7.4 selected.pkl — 최적 후보 확정

```python
{
    "source_cutoff": int,
    "target_cutoff": int,
    "selected_source_round": int,  # 최고 VAL MRR을 낸 source round
    "predecessor_checkpoint": str, # 최종 사용할 checkpoint 경로
    "best_val_MRR": float,
    ...
}
```

`build_round_data.py`의 `load_inherited_pseudo()`가 이 파일을 읽어 해당 round의 `inherited.pkl`을 로드한다.

---

## 8. Validation 방법

### 8.1 Fixed VAL Set 구성

- C3 케이스의 30%가 고정 validation set
- 각 VAL record는 다음을 포함:
  - `src`, `dst`: positive pair 인덱스
  - `hard`: hard negative 후보 인덱스 리스트 (최대 N_HARD_VAL=400개)
  - `random`: random negative 후보 인덱스 리스트 (hard + random = N_TOTAL_VAL=1,000개)

### 8.2 평가 지표

`fixed_val_metrics()` — `@torch.no_grad()`

```
per VAL record (s, d, hard, random):
  candidates = [d] + [neg1, neg2, ..., neg999]   # positive 1개 + negative 999개
  scores = model.score(z, [s]*1000, candidates)
  rank = 1 + (neg scores > pos score 수)          # 1이 최고

metrics 계산 (모든 VAL record 평균):
  MRR = mean(1/rank)
  MedianRank = median(rank)
  Hits@K = mean(rank <= K)  for K in [1, 10, 50, 100]
```

- 총 1,000개 후보 중 순위 계산 (1000-way ranking)
- 각 VAL record마다 독립적으로 순위 계산

### 8.3 Early Stopping

```
patience = 0
best_MRR = -1

for epoch in range(1, EPOCHS+1):
    ...
    if epoch % VAL_EVERY == 0:  # 10 epoch마다
        val_MRR = compute()
        if val_MRR > best_MRR:
            best_MRR = val_MRR
            save_checkpoint()
            patience = 0
        else:
            patience += 1
        if patience >= PATIENCE:  # 60번 연속 미개선
            break
```

최대 1,000 epoch, patience 60 (= 최대 600 epoch 동안 개선 없으면 종료).

---

## 9. Test 방법 (Filtered Full Ranking)

### 9.1 Test GT 구성

- **cutoff**: 2012년 스냅샷
- **test period**: 2013~2014년 소송 데이터
- test pair 수: `fixed_data/test/test_gt_2013_2014.pkl`의 `test_patent_pairs`

### 9.2 Filtered Full Ranking 절차

`test_full_ranking.py`의 `full_rank()` 함수

**Filtered Full Ranking**: 후보에서 이미 알려진 positive를 제외한 후 전체 순위 계산

```
per test pair (src, true_dst):
  blocked = {
      historical GT MP in sue edges (2012 snapshot),
      inherited pseudo MP,
      다른 test GT positive pairs with same src,
      src 자신
  }
  blocked.discard(true_dst)  # 평가 대상 true_dst는 반드시 유지

  true_score = model.score(z, src, true_dst)

  higher = 0
  for all patents in [0, N_pat) in chunks of 100,000:
      cand = [p for p not in blocked]
      scores = model.score(z, [src]*len(cand), cand)
      higher += (scores > true_score).sum()

  rank = 1 + higher
```

- **전체 특허 대상 (약 수십만~수백만 노드)**, chunk_size=100,000 단위로 배치 처리
- Filtered: 이미 알려진 positive는 경쟁에서 제외하여 rank가 억울하게 밀리는 것 방지

### 9.3 출력 지표

```
MRR = mean(1/rank)
MeanRank = mean(rank)
MedianRank = median(rank)
Hits@K for K in [1, 10, 50, 100, 500, 1000, 10000]
```

결과 저장:
- `runs/final_test_2012/test_pair_ranks.csv`
- `runs/final_test_2012/final_test_metrics.json`

---

## 10. 전체 파이프라인 실행 순서

```
[Phase 0] 사전 준비 (1회성)
  build_unified_pools.py --cutoff 2004/2006/2008/2010
  → fixed_data/unified_pools/pool_{cutoff}.pkl 생성

  (c3_data.pkl은 별도 스크립트 또는 notebook으로 이미 생성됨)

[Phase 1] Fold 2004 — R0 학습
  build_round_data.py --cutoff 2004 --target-start 2005 --target-end 2006 --round 0
  train_round.py --cutoff 2004 --round 0
  → runs/cutoff_2004/round_00/checkpoint.pt

[Phase 2] Fold 2004 — R0~R10 iterative
  for r in 0..9:
    score_case12.py --cutoff 2004 --target-start 2005 --target-end 2006 --round r
    analyze_top1.py --cutoff 2004 --round r
    select_pseudo.py --cutoff 2004 ... --round r --threshold 0.99
    build_round_data.py --cutoff 2004 ... --round r+1
    train_round.py --cutoff 2004 --round r+1

[Phase 3] Fold 2006 — Predecessor 선택
  build_inherited.py --from-year 2004 --to-year 2006 --max-round 10
  → runs/predecessor_2004_to_2006/source_round_{00..10}/inherited.pkl

  train_predecessor.py --source-cutoff 2004 --cutoff 2006 ...
  → runs/predecessor_2004_to_2006/selected.pkl  (최고 VAL MRR R 선택)

[Phase 4] Fold 2006 — R0~R10 iterative
  (Phase 1,2와 동일, 단 R0은 predecessor warm-start 자동 적용)

[Phase 5~8] Fold 2008, 2010 동일 반복
  (run_2008_to_2010.py가 2008 fold 전체 자동화)

[Phase 9] Final Test
  test_full_ranking.py
  → runs/final_test_2012/final_test_metrics.json
```

---

## 11. 하이퍼파라미터 요약

| 파라미터 | 값 | 설명 |
|----------|----|------|
| `SEED` | 42 | 전역 random seed |
| `HIDDEN` | 128 | GNN hidden 차원 |
| `DROPOUT` | 0.30 | Dropout 비율 |
| `EPOCHS` | 1,000 | 최대 학습 epoch |
| `PATIENCE` | 60 | Early stopping (VAL 검증 횟수 기준) |
| `VAL_EVERY` | 10 | Validation 주기 (epoch) |
| `LR` | 1e-3 | AdamW learning rate |
| `WEIGHT_DECAY` | 1e-4 | AdamW weight decay |
| `GRAD_CLIP` | 1.0 | Gradient clipping max norm |
| `TRAIN_RATIO` | 0.70 | C3 sup:val 분할 비율 |
| `N_HARD_TRAIN` | 2 | 학습 시 per-positive hard negative 수 |
| `N_RANDOM_TRAIN` | 3 | 학습 시 per-positive random negative 수 |
| `N_HARD_VAL` | 400 | VAL record당 hard negative 수 |
| `N_TOTAL_VAL` | 1,000 | VAL record당 전체 negative 수 (hard+random) |
| `PSEUDO_POOL_MIN_SCORE` | 75.0 | (예비 파라미터) |
| `PSEUDO_THRESHOLD` | 0.99 | Pseudo 선택 sigmoid score 임계값 |
| `TARGET_SIZE` (pool) | 2,000 | Unified pool 목표 크기 |
| `RANK_CHUNK_SIZE` | 100,000 | Full ranking 배치 크기 |
| `TRAIN_FOLDS` | [(2004,2005,2006), (2006,2007,2008), (2008,2009,2010), (2010,2011,2012)] | 학습 fold 정의 |
| `TEST_FOLD` | (2012,2013,2014) | 테스트 fold |
| `MAX_ROUNDS` | 10 | fold당 최대 iterative round 수 |

---

## 12. 주요 파일 역할 요약

| 파일 | 역할 |
|------|------|
| `config.py` | 모든 하이퍼파라미터 및 경로 상수 정의 |
| `data.py` | 스냅샷 로드, 노드 피처 빌드, 엣지 타입 정의, 가시성 마스킹 |
| `model.py` | GNN 모델 (StreamingHeteroLayer + MLPDecoder) |
| `iterative/build_round_data.py` | Round별 학습 데이터 조립 (SUP/VAL/pseudo/neg pool) + 7중 leakage 검증 |
| `iterative/train_round.py` | Round별 실제 학습 + VAL MRR 기반 early stopping |
| `iterative/score_case12.py` | 현재 체크포인트로 Case1/2 전체 스코어링 |
| `iterative/analyze_top1.py` | Top1 후보 추출 및 임계값별 분포 분석 |
| `iterative/select_pseudo.py` | threshold 기반 pseudo 선택 + 8단계 필터링 + 검증 |
| `iterative/build_inherited.py` | Source fold pseudo → target fold remap (patent ID 기준) |
| `iterative/train_predecessor.py` | 모든 source rounds 후보 학습 → VAL MRR 비교 → 최적 선택 |
| `iterative/build_unified_pools.py` | Unified hard-negative pool 최초 빌드 (B>A>fallback 우선순위) |
| `iterative/pool_manager.py` | UnifiedPoolManager (lazy-build 지원, active negative 계산) |
| `iterative/state.py` | FoldState (round 진행 중 누적 pseudo 상태 저장) |
| `run_2008_to_2010.py` | 2008 fold R0~R10 + 2010 Predecessor 비교 자동화 스크립트 |
| `test_full_ranking.py` | Final test: 2012 snapshot 기반 Filtered Full Ranking 평가 |
| `fixed_data/c3_pools/` | Fold별 고정 C3 SUP/VAL split (학습 전 1회 생성, 이후 불변) |
| `fixed_data/unified_pools/` | Fold별 hard-negative pool (학습 전 1회 생성, 이후 불변) |
| `fixed_data/test/` | Final test GT |
| `runs/` | 모든 학습 결과물 (round_data, checkpoint, metrics, pseudo 선택 결과) |
| `baseline/` | pseudo 없이 R0만 학습한 baseline 결과 |
