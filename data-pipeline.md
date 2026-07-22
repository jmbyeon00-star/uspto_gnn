# MANTF 데이터 구축 파이프라인

## src/ 폴더에 있는 파일 현황

```
src/
├── cases.csv                      # 소송 케이스 메타데이터 (81,350건)
├── patents.csv                    # 케이스별 분쟁 특허 (122,252건)
├── names.csv                      # 케이스별 원고/피고 기업명 (745,235건)
├── g_patent.tsv                   # USPTO 전체 특허 기본정보 (9,454,161건)
├── g_patent_abstract.tsv          # USPTO 전체 특허 초록 (9,454,161건)
├── g_assignee_disambiguated.tsv   # 특허별 소유 기업 (8,751,310건)
├── g_us_patent_citation.tsv       # 특허 간 인용 관계 (152,631,929건)
├── g_ipc_at_issue.tsv             # 특허 IPC 분류 코드
├── g_application.tsv              # 특허 출원 정보
├── g_inventor_not_disambiguated.tsv
├── g_foreign_priority.tsv
├── g_location_not_disambiguated.tsv
└── g_other_reference.tsv
```

---

## 파일 간 관계 및 조인 키

```
cases.csv ──(case_row_id)──► patents.csv    : 케이스 → 분쟁 특허
cases.csv ──(case_row_id)──► names.csv      : 케이스 → 원고/피고 기업명
patents.csv ──(patent)──────► g_patent.tsv  : 특허번호 → 특허 제목/기본정보
patents.csv ──(patent)──────► g_patent_abstract.tsv : 특허번호 → 초록
patents.csv ──(patent)──────► g_us_patent_citation.tsv : 특허번호 → 인용관계
```

### cases.csv 주요 컬럼
| 컬럼 | 설명 |
|------|------|
| `case_row_id` | 케이스 고유 ID (조인 키) |
| `case_number` | PACER 케이스 번호 |
| `date_filed` | 소송 제기일 |
| `date_closed` | 소송 종료일 |

### patents.csv 주요 컬럼
| 컬럼 | 설명 |
|------|------|
| `case_row_id` | cases.csv 조인 키 |
| `patent` | 특허번호 (g_patent.tsv의 patent_id와 매칭) |
| `patent_doc_type` | 특허 문서 타입 |

### names.csv 주요 컬럼
| 컬럼 | 설명 |
|------|------|
| `case_row_id` | cases.csv 조인 키 |
| `party_type` | `Plaintiff` / `Defendant` (외 180여 종) |
| `name` | 기업명 (raw, 정규화 필요) |

---

## 주요 데이터 특성 (통계)

| 항목 | 수치 |
|------|------|
| 총 소송 케이스 | 81,350건 |
| 분쟁 특허 레코드 | 122,252건 |
| 고유 분쟁 특허 수 | 45,767개 |
| 원고/피고 레코드 | 745,235건 |
| USPTO 전체 특허 | 9,454,161개 |

케이스 1건당 특허 수 분포:
- 특허 1개: 29,153건 (가장 많음)
- 특허 2개: 10,483건
- 특허 0개: 1,229건 (특허번호 미기재)

케이스 1건당 원고 수: 대부분 1~2명, 피고도 1~2명이지만 다수인 경우도 있음

---

## MANTF 모델에 필요한 파일 생성 순서

### Step 1. 소송 트리플 추출 → samples/

`cases.csv` + `patents.csv` + `names.csv` 를 `case_row_id`로 조인해서
`(원고, 피고, 특허)` 트리플 생성

**처리 이슈:**
- 원고/피고가 여럿인 케이스: 모든 조합으로 확장하거나 대표 1개 선택
- `party_type`이 `Plaintiff`, `Defendant`만 사용 (Counter Claimant 등 제외)
- 기업명 정규화 필요 (대소문자, Inc./Corp. 등 처리)
- 특허번호가 없는 케이스(1,229건) 제외

**출력:**
```
data/samples/company_index.txt    # 기업명 → 인덱스
data/samples/patent_index.txt     # 특허번호 → 인덱스
data/samples/train.csv
data/samples/val.csv
data/samples/test.csv
```

**train/val/test 분할:** `date_filed` 기준 시간순 분할 권장
- 예: ~2010 → train, 2011~2012 → val, 2013~ → test

**train/val/test CSV 형식:**
```
pos_pla_id, pos_def_id, pos_pat_id, neg_id, neg_type
```
- neg_type 0: neg_id = 다른 원고 기업명
- neg_type 1: neg_id = 다른 피고 기업명
- neg_type 2: neg_id = 다른 특허번호
- 세 타입을 동일한 수로 생성 (학습 시 0,1을 50% drop해서 1:1:2 비율)

---

### Step 2. 기업 소송 그래프 생성 → features/comp_graph.npy

소송 기록에서 원고→피고 엣지를 구성

```python
# 형식: shape (N_edges, 2) — [src_company_idx, dst_company_idx]
edges = [[company2idx[pla], company2idx[def]] for (pla, def, pat) in triples]
np.save('comp_graph.npy', np.array(edges, dtype=np.int32))
```

**파일명 규칙 참고** (`patex_percentage_wise_80_l_d_10_graph.npy`):
- `percentage_wise_80`: 전체 데이터의 80% 기간 내 소송 비율 기준으로 엣지 필터링
- `l_d_10`: 원고(l)→피고(d) 방향, 최소 10번 이상 소송한 쌍만 포함

즉 단순히 모든 소송 쌍을 엣지로 넣는 게 아니라 **소송 빈도 threshold**를 두고 필터링함

---

### Step 3. 기업 초기 특징 생성 → features/comp_feat.npy

**선택지 A (권장, 간단):** `comp_feat_path=None`으로 설정 → 랜덤 초기화, GCN이 학습

**선택지 B (논문 방식):** IPC 분류 코드 기반 기업 기술 프로파일 구성
- `g_assignee_disambiguated.tsv`로 기업별 보유 특허 목록 구성
- `g_ipc_at_issue.tsv`로 특허별 IPC 코드 추출
- 기업별 IPC 분포 벡터 → `comp_feat.npy`

```
shape: (기업 수, 특징 차원)
예: (company_num, 128)
```

---

### Step 4. 특허 BERT 임베딩 생성 → features/*.npy

**대상:** `patent_index.txt`에 있는 특허들만 (전체 USPTO 특허 X)

**소스 파일:**
- 제목: `g_patent.tsv`의 `patent_title` 컬럼
- 초록: `g_patent_abstract.tsv`의 `patent_abstract` 컬럼
- 청구항: USPTO Bulk XML에서 별도 추출 필요 (현재 src/에 없음)

**처리 방법:**
```python
from transformers import AutoTokenizer, AutoModel
import torch, numpy as np

model_name = "bert-base-uncased"  # 또는 "anferico/bert-for-patents"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert = AutoModel.from_pretrained(model_name)

def encode(texts, batch_size=32):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True,
                           max_length=512, padding=True)
        with torch.no_grad():
            out = bert(**inputs)
        all_embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())  # [CLS]
    return np.vstack(all_embs)

titles_emb = encode(title_list)    # shape: (특허 수, 768)
abstract_emb = encode(abstract_list)
claim_emb = encode(claim_list)

np.save('pat_titles.npy', titles_emb)
np.save('pat_abstracts.npy', abstract_emb)
np.save('pat_avg_claims.npy', claim_emb)
```

**청구항이 없는 경우:** 초록으로 대체하거나 zero vector

**추천 BERT 모델:** `anferico/bert-for-patents` (특허 도메인 특화)

---

### Step 5. DeepWalk 특허 인용 임베딩 생성 → features/deepwalk.npy

`g_us_patent_citation.tsv`에서 특허 인용 그래프 구성 후 DeepWalk 실행

```python
# 인용 그래프: 소송 특허들만 필터링
citation = pd.read_csv('g_us_patent_citation.tsv', sep='\t',
                       usecols=['patent_id', 'citation_patent_id'])
# 소송 관련 특허만
litigation_patents = set(patent_index)
citation_filtered = citation[
    citation['patent_id'].isin(litigation_patents) |
    citation['citation_patent_id'].isin(litigation_patents)
]
```

**DeepWalk 실행:**
```bash
pip install pecanpy
pecanpy --input citation_graph.edgelist --output deepwalk.emb \
        --workers 8 --dimensions 100 --walk-length 80 --num-walks 10
```

**출력:**
```
shape: (특허 수, 100)  ← 모델의 MLP 입력이 3*768+100=2404 이므로 100 고정
```

---

## 최종 파일 구조

```
data/
├── samples/
│   ├── company_index.txt          # 기업명,1 (한 줄 = 하나의 기업)
│   ├── patent_index.txt           # 특허번호,1 (한 줄 = 하나의 특허)
│   ├── train.csv                  # pos/neg 트리플 쌍
│   ├── val.csv
│   └── test.csv
└── features/
    ├── comp_graph.npy             # shape: (엣지 수, 2)
    ├── comp_feat.npy              # shape: (기업 수, feat_dim) — 선택
    ├── pat_titles.npy             # shape: (특허 수, 768)
    ├── pat_abstracts.npy          # shape: (특허 수, 768)
    ├── pat_avg_claims.npy         # shape: (특허 수, 768)
    └── patent_deepwalk_embedding.npy  # shape: (특허 수, 100)
```

---

## 주요 이슈 및 결정사항

### 1. 케이스당 원고/피고 여러 명 처리
- **옵션 A:** 모든 (원고, 피고) 조합을 트리플로 확장 → 데이터 폭발 위험
- **옵션 B (권장):** 원고 중 첫 번째, 피고 중 첫 번째만 대표로 사용

### 2. 케이스당 특허 여러 개 처리
- 각 특허마다 트리플 생성 → (원고, 피고, 특허1), (원고, 피고, 특허2), ...

### 3. 기업명 정규화
`names.csv`의 기업명은 raw 텍스트라 동일 기업이 다르게 표기됨
- "Apple Inc." / "Apple, Inc." / "APPLE INC" 등
- 최소한 소문자 변환 + 구두점 제거 필요
- 더 정확하려면 string matching 또는 외부 entity resolution 필요

### 4. 청구항 데이터
현재 `src/`에 없음. USPTO Bulk Data XML에서 별도 다운로드 필요
없으면 `pat_avg_claims.npy`를 `pat_abstracts.npy`로 대체 가능

### 5. train/val/test 분할 비율
논문 파일명 `percentage_wise_train_80`을 보면 80:10:10 또는 80:20 으로 추정
시간순 분할을 권장 (랜덤 분할 시 미래 소송으로 과거 예측하는 leakage 발생)
