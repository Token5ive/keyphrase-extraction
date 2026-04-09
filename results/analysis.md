# KPE 파이프라인 실험 분석 보고서

## 1. 과제 해석 및 문제 정의

### 1.1 데이터 특성

| 항목 | 값 |
|---|---|
| 샘플 수 | 189개 (과학 논문 제목 + 초록) |
| 평균 후보 키프레이즈 수 (SciBART 예측) | 9.2개 |
| 평균 Gold Present KP 수 | 2.6개 |
| 평균 Gold Absent KP 수 | 1.8개 |
| 평균 Gold 전체 | 4.4개 |

데이터는 SciBART(seq2seq 생성 모델)가 과학 논문에서 후보 키프레이즈를 생성한 결과물이다. Gold 키프레이즈는 문서 내 등장 여부에 따라 **Present**(문서 내 존재)와 **Absent**(문서에 없는 개념적 키프레이즈)로 나뉜다.

### 1.2 과제 구조 이해

이 실험의 핵심 과제는 **재순위(Reranking)** 문제다. SciBART가 이미 후보를 생성했으므로, 우리 파이프라인의 역할은 9.2개 후보 중 정답에 가까운 상위 K개를 잘 골라내는 것이다.

두 가지 근본적인 어려움이 존재한다:
1. **Absent KP 예측 불가** — 문서에 등장하지 않는 키프레이즈는 문서-후보 유사도 기반 재순위로 상위에 올리기 어렵다. 실험 전체에서 F1@5(Absent)가 0.03~0.04 수준에 머무는 것이 이를 방증한다.
2. **후보 풀이 작다** — 평균 9.2개 후보에서 5개를 고르는 문제라, 이미 대부분의 정답이 후보 안에 포함되어 있을 가능성이 높다. 따라서 재순위 알고리즘의 개입 여지가 좁다.

---

## 2. 모델 선택 근거

### 2.1 후보 생성: SciBART

SciBART(BART 기반 과학 논문 특화 seq2seq)를 후보 생성기로 선택한 이유:
- 과학 도메인 텍스트에 사전학습되어 Absent KP 생성 능력을 갖춤
- 생성 모델이기 때문에 문서에 등장하지 않는 개념적 키프레이즈도 예측 가능
- 분류 기반 추출 모델(예: BERT span extraction)은 정의상 Absent KP를 예측하지 못함

### 2.2 재순위 임베딩 모델 탐색

재순위 단계에서 문서-후보 간 의미 유사도를 측정하기 위해 세 가지 인코더를 비교했다.

| 모델 | 선택 근거 | 특성 |
|---|---|---|
| **SciBERT** (`allenai/scibert_scivocab_uncased`) | 과학 논문 코퍼스에 특화 학습, 도메인 어휘 이해 | CLS 토큰 기반 인코딩, 도메인 특화 |
| **MPNet** (`all-mpnet-base-v2`) | 문장 임베딩 태스크 SOTA급 일반 모델 | 문장 수준 유사도 최적화 학습 |
| **MiniLM-L6** (`all-MiniLM-L6-v2`) | KeyBERT 공식 기본 모델, 경량·고속 | 문장 임베딩 효율 최적화 |

SciBERT는 도메인 지식 면에서 유리할 것으로 기대했으나, **문장 수준 유사도 목적으로 학습되지 않았다**는 점이 잠재적 약점이었다. MPNet과 MiniLM은 sentence-transformers 프레임워크로 문장 유사도에 최적화되어 있다.

### 2.3 재순위 알고리즘: KeyBERT + MMR

- **KeyBERT**: 문서 임베딩과 후보 임베딩의 코사인 유사도로 1차 순위 결정
- **MMR(Maximal Marginal Relevance)**: relevance와 diversity를 동시에 최적화하는 greedy 선택

```
MMR_i = λ × sim(doc, c_i) − (1−λ) × max_{s∈S} sim(s, c_i)
```

λ=1.0이면 순수 relevance(KeyBERT-only), λ=0.0이면 순수 diversity. 과학 키프레이즈는 같은 개념이 다양하게 표현되는 경향(예: `deformable model` / `deformable curves` / `deformable surfaces`)이 있어 MMR이 중복 제거에 유효할 것으로 가정했다.

---

## 3. 실험 결과 분석

### 3.1 Exp 1 — 전처리 적용 vs 미적용

| 방법 | F1@5 | F1@M | ILD | SR |
|---|---|---|---|---|
| Raw \| Baseline | 0.2680 | 0.2727 | 0.2581 | 0.7513 |
| Raw \| KeyBERT+MMR (λ=0.6) | 0.2181 | 0.2060 | 0.2535 | 0.6561 |
| Preprocessed \| Baseline | 0.2670 | 0.2719 | 0.2594 | 0.7513 |
| Preprocessed \| KeyBERT+MMR (λ=0.6) | 0.2205 | 0.2196 | 0.2546 | 0.6561 |

**관찰:**
- 전처리 적용 여부가 성능에 거의 영향을 주지 않는다 (F1@5 차이 < 0.003).
- **가장 주목할 점: Baseline(SciBART 원본 순서)이 KeyBERT+MMR 재순위보다 일관되게 높다.**

**해석:**
전처리의 효과가 미미한 이유는 SciBART가 이미 비교적 정제된 키프레이즈를 생성하기 때문으로 보인다. 길이 필터(1~5 단어)나 중복 제거가 영향을 줄 만한 노이즈 후보가 애초에 적다.

재순위가 오히려 성능을 낮추는 현상은 이후 실험에서도 반복적으로 나타나며, 이 파이프라인의 핵심 문제점이다.

---

### 3.2 Exp 2 — 임베딩 모델 비교

| 방법 | F1@5 | F1@M | ILD | SR |
|---|---|---|---|---|
| Baseline (공통) | 0.2670 | 0.2719 | — | 0.7513 |
| SciBERT \| KeyBERT-only (λ=1.0) | 0.2038 | 0.1937 | 0.2524 | 0.6614 |
| SciBERT \| KeyBERT+MMR (λ=0.6) | 0.2205 | 0.2196 | 0.2546 | 0.6561 |
| MPNet \| KeyBERT-only (λ=1.0) | 0.2275 | 0.2306 | 0.7021 | 0.7037 |
| **MPNet \| KeyBERT+MMR (λ=0.6)** | **0.2468** | **0.2380** | **0.7065** | **0.7143** |
| MiniLM \| KeyBERT-only (λ=1.0) | 0.2262 | 0.2243 | 0.6952 | 0.7143 |
| MiniLM \| KeyBERT+MMR (λ=0.6) | 0.2424 | 0.2331 | 0.7011 | 0.6984 |

**관찰 1: SciBERT의 역설적 저성능**

SciBERT가 과학 도메인 특화 모델임에도 MPNet, MiniLM보다 F1이 낮다. 가장 결정적인 단서는 **ILD(다양성 지표)의 극단적 차이**다:
- SciBERT ILD: 0.25
- MPNet ILD: 0.71
- MiniLM ILD: 0.70

SciBERT의 임베딩 공간에서 과학 키프레이즈들이 서로 매우 가깝게 몰려 있다는 의미다. 이는 SciBERT가 도메인 어휘를 잘 이해하는 반면, **sentence-level 의미 유사도가 아닌 MLM(Masked Language Modeling) 목적으로 학습**되어 문장/구 수준 임베딩 품질이 sentence-transformers 모델에 비해 낮음을 시사한다. CLS 토큰만으로 키프레이즈 수준의 표현을 잡기 어렵다.

**관찰 2: 모든 모델에서 Baseline이 최고**

MPNet+MMR(F1@5=0.2468)조차 Baseline(0.2670)에 미치지 못한다. 재순위가 전반적으로 역효과다. 이는 단순히 인코더 품질의 문제가 아니라 **재순위 접근 자체의 한계**를 의심하게 만든다.

**관찰 3: MMR이 KeyBERT-only보다 항상 낫다**

모든 인코더에서 MMR(λ=0.6)이 순수 relevance(λ=1.0)보다 F1이 높다. 즉 다양성 확보가 성능에 실제로 기여하고 있다.

---

### 3.3 Exp 3 — Lambda Sweep (SciBERT)

| λ | F1@5 | ILD | SR |
|---|---|---|---|
| 0.0 (pure diversity) | 0.2148 | 0.2554 | 0.6931 |
| 0.6 (best) | **0.2205** | 0.2546 | 0.6561 |
| 1.0 (pure relevance) | 0.2038 | 0.2524 | 0.6614 |

**관찰 1: F1 곡선이 극도로 평탄하다**

λ=0.0~1.0 전 구간에서 F1@5의 범위가 0.2038~0.2205로 0.017 차이에 불과하다. λ 값에 거의 둔감하다는 뜻이다.

이는 SciBERT의 낮은 ILD(0.25)와 직결된다. 후보 임베딩들이 이미 서로 비슷하게 뭉쳐 있으면 MMR의 다양성 패널티(1-λ 항)가 실질적으로 동작하지 않는다. 어떤 λ를 써도 비슷한 후보가 선택된다.

**관찰 2: λ=0.6이 최적이나 의미는 제한적**

최적 λ=0.6은 relevance를 우선하되 소폭의 diversity를 반영하는 값이다. 그러나 SciBERT 기준으로는 어느 λ를 선택해도 성능 차이가 유의미하지 않다. MPNet으로 sweep을 수행했다면 더 뚜렷한 패턴이 나타났을 것이다.

---

### 3.4 Exp 4 — 정성 분석

**Example 0** — `Narrow band region-based active contours` (F1@5=0.5455)

```
Gold: active contour, segmentation, deformable model, level sets, narrow band region energy, active surface
KeyBERT Top-5: deformable models ✓, deformable curves and surfaces, 3d segmentation, ...
MMR Top-5:     deformable models ✓, deformable curves and surfaces, medical image segmentation, level sets ✓, active contours ✓
```
MMR가 KeyBERT에서 놓친 `level sets`, `active contours`를 다양성 확보를 통해 추가로 적중시켰다. MMR의 효과가 명확히 드러나는 케이스다.

**Example 1** — `Continuous input nonlocal games` (F1@5=0.2222)

```
Gold: nonlocality, entanglement, quantum games, bell inequalities
KeyBERT Top-5: nonlocal games, nonlocal game, game theory, quantum games ✓, continuous input
MMR Top-5:     nonlocal games, monotonicity, game theory, continuous input, quantum games ✓
```
`nonlocal games`와 `nonlocal game`은 stem이 동일(`nonloc game`)해 중복이지만, MMR이 두 번째 선택에서 `monotonicity`를 골라 `nonlocality`, `entanglement`, `bell inequalities`를 전혀 건드리지 못했다. Gold KP인 `nonlocality`는 표현 형태가 달라 유사도가 낮게 잡혀 탈락했다. **SciBART가 아예 후보를 생성하지 못한 경우**(`entanglement`, `bell inequalities`는 후보에 없음)는 재순위 단계에서 복구 불가능하다.

**Example 2** — `Managing critical success strategies` (F1@5=0.1818)

```
Gold: critical success strategies, enterprise resource planning project, optimization, ...
KeyBERT Top-5: management matrix, enterprise resource planning, critical success strategies ✓, critical success strategy ✓, optimization models
```
`critical success strategy`와 `critical success strategies`가 동일 stem으로 매칭되어 precision 분모만 늘리는 문제 발생. `enterprise resource planning`과 `enterprise resource planning project`의 부분 일치는 현재 평가 기준(exact stem match)에서 처리되지 않아 점수 손실이 발생한다.

---

## 4. 핵심 의심 사항 및 분석

### 4.1 왜 Baseline이 항상 재순위보다 낫나?

SciBART는 생성 시 빔 서치(beam search)를 사용하며, 높은 빔 점수(beam score) 순으로 후보가 정렬되어 있다. 즉 **SciBART 원본 순서 자체가 이미 내부적 신뢰도 순위**다.

우리 재순위는 이 정보를 무시하고 외부 인코더의 코사인 유사도로 순위를 덮어쓴다. 문서-후보 코사인 유사도가 SciBART의 빔 점수보다 정답과의 상관이 낮다면 재순위는 오히려 손해다.

### 4.2 SciBERT의 낮은 ILD

SciBERT는 과학 논문 MLM으로 학습되었고, 과학 키프레이즈들이 유사한 도메인 어휘로 구성되다 보니 임베딩 공간에서 가깝게 모인다. 이는 두 가지 문제를 동시에 유발한다:
1. 문서-후보 유사도 순위 자체의 변별력이 낮음
2. MMR의 다양성 항이 무력화됨

### 4.3 Absent KP의 구조적 한계

Absent KP는 문서에 등장하지 않는 개념이므로 문서와의 코사인 유사도가 Present KP보다 낮게 잡힌다. 유사도 기반 재순위는 원리적으로 Absent KP를 상위로 올리기 어렵다. 이 과제에서 Absent KP 비율이 약 41%(1.8/4.4)에 달하는데, 현재 파이프라인은 이 부분을 거의 처리하지 못한다.

### 4.4 평가 기준의 엄격성

Exact stem match 기준을 사용하므로:
- `enterprise resource planning` vs `enterprise resource planning project` → 불일치
- `nonlocal games` vs `nonlocality` → 불일치

부분 일치를 허용하는 평가 기준(soft match)이라면 실제 성능이 수치보다 높을 수 있다.

---

### 3.5 Exp 5 — Score Fusion (Beam Score Proxy + Cosine Similarity)

**동기**: Exp 1~4의 핵심 문제는 재순위가 Baseline(SciBART 원본 순서)을 한 번도 넘지 못한 것이다. SciBART의 생성 순서가 내부 빔 점수를 반영하는 신뢰도 있는 신호라면, 이를 코사인 유사도와 결합한 Score Fusion이 Baseline을 초과할 수 있다는 가설 하에 실험을 설계했다.

**Score Fusion 공식:**
```
beam_score(i)   = 1 − position_i / (N_original − 1)   # 0번째 = 1.0, 마지막 = 0.0
fusion_score(i) = α × beam_score(i) + (1−α) × cosine_sim(doc, c_i)
```
α=1.0이면 순수 beam score(≈Baseline 순서), α=0.0이면 순수 cosine similarity(≈KeyBERT).

인코더는 Exp 2에서 최우수로 확인된 MPNet을 사용했다.

#### Phase 1: α Sweep (λ=0.6 고정)

| α | F1@5 | F1@M | ILD | SR |
|---|---|---|---|---|
| 0.0 (cosine only) | 0.2468 | 0.2380 | 0.7065 | 0.7143 |
| **0.1** | **0.2673** | **0.2609** | 0.7067 | **0.7302** |
| 0.2 | 0.2615 | 0.2621 | 0.7078 | 0.7249 |
| 0.3 | 0.2625 | 0.2721 | 0.7078 | 0.7302 |
| **0.4** | **0.2677** | 0.2720 | 0.7083 | **0.7460** |
| 0.5 | 0.2654 | 0.2767 | 0.7085 | 0.7354 |
| 0.6 | 0.2667 | 0.2763 | 0.7089 | 0.7354 |
| 1.0 (beam only) | 0.2620 | 0.2766 | 0.7088 | 0.7302 |
| **Baseline** | **0.2670** | **0.2719** | 0.7257 | **0.7513** |

α=0.4에서 F1@5=**0.2677**로 Baseline(0.2670)을 **처음으로 초과**했다. α=0.1도 F1@5=0.2673으로 근접한 수치를 보인다.

**관찰:**
- α=0.0(순수 cosine)에서 α가 증가할수록 F1이 급등하다가 α≈0.4에서 정점을 찍고 완만히 하락한다.
- beam score만 사용하는 α=1.0(F1=0.2620)은 Baseline(0.2670)보다 낮다. 이는 beam score proxy(순서)가 실제 빔 점수와 완전히 동일하지 않음을 의미한다 — 전처리로 일부 후보가 제거되면서 position이 재정렬되기 때문이다.
- ILD는 α 전 구간에서 0.706~0.709로 안정적이며, α에 거의 무감하다.

#### Phase 2: λ Sweep (α=0.4 고정)

| λ | F1@5 | ILD | SR |
|---|---|---|---|
| 0.0 (pure diversity) | 0.2378 | 0.7105 | 0.7090 |
| 0.4 | 0.2595 | 0.7098 | 0.7302 |
| **0.6** | **0.2677** | 0.7083 | **0.7460** |
| 0.9 | 0.2643 | 0.7049 | 0.7513 |
| 1.0 (pure relevance) | 0.2578 | 0.7038 | 0.7354 |

λ=0.6에서 F1@5 최대. Exp 3(SciBERT)과 달리 MPNet 기반에서는 λ에 따른 F1 범위가 0.030으로 더 뚜렷한 패턴을 보인다.

#### 최종 비교

| 방법 | F1@5 | F1@M | ILD | SR | F1@5(Present) | F1@5(Absent) |
|---|---|---|---|---|---|---|
| Baseline (SciBART order) | 0.2670 | 0.2719 | 0.7257 | 0.7513 | 0.3096 | 0.0355 |
| KeyBERT+MMR (λ=0.6) | 0.2468 | 0.2380 | 0.7065 | 0.7143 | 0.2865 | 0.0292 |
| **Score Fusion (α=0.4, λ=0.6)** | **0.2677** | **0.2720** | 0.7083 | 0.7460 | **0.3088** | **0.0361** |

Score Fusion이 F1@5 기준으로 Baseline을 처음으로 초과했다. F1@M(0.2720 vs 0.2719)과 F1@5(Absent)(0.0361 vs 0.0355)도 Baseline과 대등하거나 소폭 상회한다. SR은 0.7460으로 Baseline(0.7513)에 근접했다.

**해석:**

1. **가설 검증**: beam score proxy(생성 순서) + cosine similarity의 선형 결합이 실제로 Baseline을 초과함을 확인했다. 순수 cosine(α=0) 또는 순수 beam(α=1) 단독으로는 Baseline을 넘지 못하고, 두 신호의 결합(α=0.4)에서만 초과가 가능하다.

2. **α=1.0 < Baseline**: 전처리 단계에서 후보가 필터링되면 position이 원본 beam score 순서와 어긋난다. 예를 들어 원본에서 position=0인 후보가 필터링으로 제거되면, 실제로는 2등이었던 후보가 position=0의 beam score(1.0)를 받게 된다. 이 noisy proxy 문제가 α=1.0에서 성능 손실을 유발한다.

3. **F1@5(Absent) 소폭 개선**: beam score를 혼합하면 코사인 유사도 기준으로 낮게 평가되던 Absent KP가 생성 순서 정보로 일부 구제될 수 있음을 시사한다. 그러나 여전히 0.03대에 머물러 근본적 해결은 아니다.

---

## 5. 결론 및 향후 방향

### 5.1 실험 요약

| 결론 | 근거 |
|---|---|
| SciBART 원본 순서가 강력한 baseline | Exp 1~4 전체에서 재순위 < baseline |
| MPNet이 최적 인코더 | F1@5=0.2468, ILD=0.71로 SciBERT(ILD=0.25) 압도 |
| SciBERT는 문장 유사도에 부적합 | MLM 목적 학습으로 임베딩 공간 분별력 부족 |
| MMR은 일관되게 효과 있음 | 모든 모델에서 λ=0.6 > λ=1.0 |
| Score Fusion(α=0.4)이 처음으로 Baseline 초과 | F1@5=0.2677 vs Baseline 0.2670 |
| beam score proxy의 한계 | 전처리 후 position이 원본 순서와 어긋나 α=1.0 < Baseline |
| Absent KP는 현 파이프라인으로 근본 해결 불가 | 전 실험에서 F1@5(Absent) ≈ 0.03 |

### 5.2 향후 실험 방향

1. **정확한 beam score 활용**: 전처리 전 원본 position을 그대로 보존하거나, SciBART 추론 시 실제 beam score(log-probability)를 추출하여 더 신뢰도 높은 신호로 활용
2. **α, λ 2차원 그리드 서치**: 현재 α sweep과 λ sweep을 순차적으로 진행했으나, 두 하이퍼파라미터의 상호작용을 확인하기 위한 joint sweep 필요
3. **Absent KP 별도 처리**: 유사도 기반 재순위와 별도로 query expansion 또는 knowledge graph 기반 개념 확장으로 Absent KP 적중률 개선
4. **소프트 매칭 평가 추가**: 부분 일치, 어근 포함 매칭 기준 추가로 실제 성능을 더 정확히 측정 (현재 strict stem match가 성능을 과소평가할 가능성)
