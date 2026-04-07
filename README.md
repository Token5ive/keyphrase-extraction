# DiverseKey: Semantic Keyphrase Generation & Diversity Optimization

> 과학 문헌의 핵심 의미 추출 및 중복 제거를 통한 고품질 키프레이즈 생성 프로젝트

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co)

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|---|---|
| **기간** | 2026.04.02 ~ 2026.04.10 (9일) |
| **인원** | 5명 (신준수, 손중영, 심소민, 이지선, 장동욱) |
| **목표** | 학술 논문 제목·초록 → Present/Absent Keyphrase 생성 + Semantic Ranking + Diversity 최적화 |

---

## 1. 문제 정의

### Task 소개

Keyphrase Extraction/Generation은 학술 문서의 핵심 개념을 대표하는 단어·구(phrase)를 자동으로 예측하는 NLP 과제다. 정보 검색, 문서 요약, 자동 태깅 등 다양한 다운스트림 태스크의 기반 기술로 활용된다.

### 해결 문제

본 프로젝트는 다음 네 가지를 동시에 해결한다.

```
1. Present KP  — 문서 내 실제로 등장하는 핵심구 추출 (Extractive)
2. Absent KP   — 문서에 없지만 문서를 대표할 수 있는 개념 생성 (Abstractive)
3. Ranking     — 중요한 개념일수록 상위에 배치 (Semantic Ranking)
4. Diversity   — 의미적 중복 제거로 정보 밀도 극대화 (MMR 기반)
```

### 핵심 도전 과제

세 요소가 서로 **트레이드오프** 관계에 있다는 점이 이 문제를 어렵게 만든다.

| 충돌 요소 | 이유 |
|---|---|
| Present vs Absent | 추출은 정밀도에 유리, 생성은 재현율에 유리 |
| Ranking vs Diversity | 상위 ranked KP들은 의미적으로 유사할 가능성이 높음 |
| Recall vs Precision | Absent KP를 많이 생성할수록 Recall↑, Noise↑ |

### 응용 가치

- **RAG 시스템** — 검색 쿼리 생성 및 문서 인덱싱 핵심 기술
- **정보 검색(IR)** — 학술 문헌 발견 가능성(Discoverability) 증대
- **자동 태깅** — 대규모 문헌 데이터베이스 자동 분류

---

## 2. 배경 및 관련 연구

### 방법론 발전 흐름

이 분야는 각 모델이 이전 모델의 한계를 해결하며 발전해왔다.

```
통계 기반 (TF-IDF, TextRank)
    ↓ 한계: Absent KP 생성 불가
CopyRNN — Meng et al., ACL 2017
    ↓ 한계: 순서 편향, 중복 생성
One2Seq → CatSeq/CatSeqD — Yuan et al., ACL 2020
    ↓ 한계: Set이 아닌 Sequence라 순서 편향 잔존
One2Set — Ye et al., ACL 2021
    ↓ 한계: Ranking 없음, ∅ 토큰 과대추정
SimCKP — Choi et al., EMNLP Findings 2023
    ↓ 한계: Absent F1 낮음, Diversity 암묵적
One2Set + LLM — Shao et al., EMNLP 2024  ← 현재 SOTA
```

### 핵심 참고 논문

| # | 논문 | 핵심 기여 | 우리 과제 연관성 |
|---|---|---|---|
| 1 | Xie et al., IPM 2023 — *From Statistical Methods to Deep Learning* (Survey) | 167개 논문 통일 조건 재비교, 패러다임 분류 체계 | 분야 전체 지도, 용어 정의 |
| 2 | Meng et al., ACL 2017 — *Deep Keyphrase Generation* (CopyRNN) | Seq2Seq + Copy Mechanism, KP20K 데이터셋 공개 | Absent KP 생성의 시작점, KP20K 표준 확립 |
| 3 | Ye et al., ACL 2021 — *One2Set* | Set 기반 병렬 생성, Bipartite Matching | Diversity의 구조적 해결, 순서 편향 제거 |
| 4 | Choi et al., EMNLP Findings 2023 — *SimCKP* | Contrastive Learning 기반 Unified Reranker | Semantic Ranking 직접 구현, 우리 Reranker 설계 기반 |
| 5 | Shao et al., EMNLP 2024 — *One2Set + LLM* | OT Assignment + Generate-then-Select | 현재 SOTA, Diversity 지표 정량화 방법 제시 |
| 6 | Thomas & Vajjala, NAACL Findings 2024 — *Diversity Heads* | Orthogonal Regularization으로 Absent KP 다양성 | Absent KP Diversity 아키텍처 수준 보장 |
| 7 | Yuan et al., ACL 2020 — *One2Seq (CatSeq/CatSeqD)* | Semantic Coverage + Orthogonal Regularization | Diversity 기법의 원형, catSeqD가 우리 diversity 설계 참고 |
| 8 | Wu et al., EMNLP 2023 — *Rethinking Model Selection & Decoding* | PLM 선택과 Decoding 전략의 체계적 비교 | Backbone 선택(SciBART), Beam size 결정 근거 |
| 9 | Carbonell & Goldstein, SIGIR 1998 — *MMR* | Relevance-Diversity Trade-off 공식화 | 우리 Diversity 필터의 직접 구현체 |
| 10 | Hulth, EMNLP 2003 — *Inspec Dataset* | 전문 인덱서 기반 keyphrase 어노테이션 | Out-of-domain 평가 데이터셋 원본 논문 |

---

## 3. 제안 방법: Two-Stage Pipeline

단일 생성 모델의 **Recall-Precision 트레이드오프** 한계를 극복하기 위해 "대량 생성 → 의미 기반 필터링" 전략을 채택했다.

```
[입력]
title + " . " + abstract
        ↓
[Stage 1: Candidate Generation]
SciBART (Beam Search, num_beams=50, num_return=20)
        ↓ 후보 20개 (Present + Absent 혼재)
[후보 후처리]
길이 필터링 (1~6 단어) → 중복 제거 → Present/Absent 분리
        ↓
[Stage 2: MMR Diversity Reranking]
SciBERT로 문서·후보 임베딩 → Cosine Similarity 계산
        ↓ MMR 공식 적용
[최종 출력]
Top-5 Keyphrases (중요도 순, 중복 최소화)
```

### Stage 1 — Candidate Generation (SciBART)

과학 도메인 특화 SciBART를 활용해 풍부한 후보군을 확보한다.

- Beam Search (size=20~50)로 Over-generation → Recall 극대화
- Copy Mechanism으로 Present KP 자동 처리
- Seq2Seq로 Absent KP 생성

### Stage 2 — MMR Diversity Reranking

**SciBERT** 임베딩 기반으로 문서-KP 유사도를 계산하고, MMR 공식으로 최종 선택한다.

$$MMR = \arg \max_{D_i \in R \setminus S} \left[\lambda \cdot Sim_1(D_i, Q) - (1-\lambda) \max_{D_j \in S} Sim_2(D_i, D_j)\right]$$

| 기호 | 의미 |
|---|---|
| $D_i$ | 후보 Keyphrase |
| $Q$ | 원본 문서 |
| $S$ | 이미 선택된 KP 집합 |
| $\lambda$ | Relevance-Diversity 균형 파라미터 (0.5~0.7 튜닝) |

---

## 4. 데이터셋 및 전처리

### 데이터셋

| Dataset | 역할 | 도메인 | 규모 | 특성 |
|---|---|---|---|---|
| **KP20K** | 학습 + In-domain 평가 | 컴퓨터과학 (CS) | 530K (train) / 20K (test) | 저자 지정 키워드, Absent ~37% |
| **Inspec** | Out-of-domain 평가 | CS/IT 전반 | 500 (test) | 전문 인덱서 지정, Absent ~22% |

> **왜 KP20K→Inspec이 중요한가?**
> KP20K는 저자(author-assigned), Inspec은 전문 인덱서(indexer-assigned) 어노테이션으로 스타일이 다르다. 도메인 갭을 극복하는 능력이 모델의 실질적 일반화 성능을 보여준다.

### 전처리 파이프라인

전처리는 3개 지점에서 수행된다.

**지점 1 — KeyBART 입력 전처리**

```python
def preprocess_input(sample):
    text = sample["title"].lower() + " . " + sample["abstract"].lower()
    text = re.sub(r'\d+', '<digit>', text)   # 숫자 치환
    text = re.sub(r'[^\w\s\-]', ' ', text)   # 특수문자 제거 (하이픈 유지)
    return re.sub(r'\s+', ' ', text).strip()
```

**지점 2 — 후보 후처리**

```python
# 필터 기준
# - 길이: 1~6 단어
# - 중복: exact match 제거
# - Present/Absent 분리: 원문 substring 포함 여부
```

**지점 3 — 평가용 정답 KP 처리**

```python
# Porter Stemmer 적용 (형태 변형 통일)
# "neural networks" → "neural network"로 통일해 공정한 F1 측정
```

---

## 5. 실험 설계 및 평가

### 실험 그룹

| 모델 | 설명 | 목적 |
|---|---|---|
| **M0** | BART-base (Baseline) | 파인튜닝 없는 기준선 |
| **M1** | SciBART-base | 도메인 특화 효과 측정 |
| **M2** | SciBART + MMR Reranker | 제안 방법 (Proposed) |

### 평가 지표

**품질 지표**

| 지표 | 설명 |
|---|---|
| $F_1@5$ | 상위 5개 예측 기준 F1 |
| $F_1@10$ | 상위 10개 예측 기준 F1 |
| $F_1@M$ | 예측 개수 기준 F1 (가변) |

> Present / Absent 분리 보고 — Porter Stemmer 적용 후 Macro-average

**다양성 지표**

| 지표 | 설명 |
|---|---|
| Redundancy Rate | 예측 KP 간 중복 비율 |
| Avg. Pairwise Similarity | 예측 KP 간 평균 임베딩 유사도 |

### Ablation Study 설계

| 실험 | 변수 | 목적 |
|---|---|---|
| A | Reranker 유무 | Reranker 기여 측정 |
| B | Beam size (10 / 20 / 50) | 후보 수 영향 분석 |
| C | λ 값 (0.3 / 0.5 / 0.7) | Diversity-Relevance 균형 최적화 |
| D | BART vs SciBART | 도메인 특화 효과 분리 |

---

## 6. 프로젝트 구조

```
DiverseKey/
├── data/
│   ├── kp20k/                  # KP20K 데이터
│   └── inspec/                 # Inspec 데이터
├── preprocessing/
│   ├── document_preprocessor.py   # 입력 전처리
│   ├── candidate_postprocessor.py # 후보 후처리
│   ├── gold_preprocessor.py       # 평가용 정답 처리
│   └── pipeline.py                # 통합 파이프라인
├── models/
│   ├── generator.py            # SciBART 생성 모듈
│   └── reranker.py             # MMR Reranker
├── evaluation/
│   ├── metrics.py              # F1@5, F1@M 계산
│   └── diversity_metrics.py    # Diversity 지표
├── experiments/
│   ├── run_baseline.py         # M0, M1 실험
│   └── run_proposed.py         # M2 실험
├── notebooks/
│   └── EDA.ipynb               # 데이터 분석
├── requirements.txt
└── README.md
```

---

## 7. 마일스톤

| 날짜 | 내용 |
|---|---|
| **04/02** | 환경 구축, EDA (Present/Absent 비율, KP 길이 분포, Vocab 겹침 분석) |
| **04/03** | 주제 확정, 논문 조사, 역할 분담 |
| **04/04~05** | 전처리 파이프라인 구축, SciBART 파인튜닝 시작 |
| **04/06~07** | MMR Reranker 구현, 중간 코드 리뷰 |
| **04/08** | 코드 리팩토링, 파이프라인 통합, λ 하이퍼파라미터 튜닝 |
| **04/09** | 정량/정성 분석, Ablation Study, 발표자료 제작 |
| **04/10** | 최종 발표 및 성과 공유 |

---

## 8. 역할 분담

### 데이터 파트 (신준수, 장동욱, 손중영)

- KP20K / Inspec EDA (통계 분석, 시각화)
- 전처리 파이프라인 구축 (3단계 전처리)
- Present/Absent 분류 및 평가 데이터 구성
- F1 평가 코드 구현 (Porter Stemmer 적용)

### 모델링 파트 (심소민, 이지선)

- SciBART KP20K 파인튜닝
- SciBERT 임베딩 추출
- MMR Reranking 구현
- 하이퍼파라미터 최적화 (λ, Beam size)

---

## 9. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/your-repo/diversekey.git
cd diversekey

# 의존성 설치
pip install -r requirements.txt

# NLTK 데이터 다운로드
python -c "import nltk; nltk.download('punkt')"
```

**requirements.txt**

```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
datasets>=2.12.0
nltk>=3.8.0
scikit-learn>=1.3.0
keybert>=0.7.0
spacy>=3.6.0
matplotlib>=3.7.0
numpy>=1.24.0
```

---

## 10. 참고 문헌

```
[1] Xie et al. (2023). From Statistical Methods to Deep Learning,
    Automatic Keyphrase Prediction: A Survey. IPM 60(4).

[2] Meng et al. (2017). Deep Keyphrase Generation. ACL.

[3] Ye et al. (2021). One2Set: Generating Diverse Keyphrases as a Set. ACL.

[4] Choi et al. (2023). SimCKP: Simple Contrastive Learning of
    Keyphrase Representations. EMNLP Findings.

[5] Shao et al. (2024). One2Set + Large Language Model:
    Best Partners for Keyphrase Generation. EMNLP.

[6] Thomas & Vajjala (2024). Improving Absent Keyphrase Generation
    with Diversity Heads. NAACL Findings.

[7] Yuan et al. (2020). One Size Does Not Fit All: Generating and
    Evaluating Variable Number of Keyphrases. ACL.

[8] Wu et al. (2023). Rethinking Model Selection and Decoding for
    Keyphrase Generation. EMNLP.

[9] Carbonell & Goldstein (1998). The Use of MMR for
    Diversity-Based Reranking. SIGIR.

[10] Hulth (2003). Improved Automatic Keyword Extraction
     Given More Linguistic Knowledge. EMNLP.
```
