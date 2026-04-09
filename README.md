# Keyphrase Extraction

KeyBERT + MMR 기반 키프레이즈 재순위(reranking) 파이프라인.

SciBART가 생성한 후보 키프레이즈를 SciBERT 임베딩으로 재정렬하고, MMR(Maximal Marginal Relevance)로 다양성을 확보합니다.

## 파이프라인

```
predictions.json (SciBART 후보)
        ↓
Preprocessor — 노이즈 제거, 길이 필터, 중복 제거
        ↓
Embedder (SciBERT / MiniLM / MPNet) — 문서·후보 임베딩
        ↓
Reranker — 코사인 유사도 → MMR 계산 후 재순위
        ↓
Evaluator — P@K, R@K, F1@K, ILD, SR
```

## 파일 구조

```
keyphrase-extraction/
├── main.py            # 실험 실행 스크립트
├── requirements.txt
├── data/
│   └── predictions.json
└── src/
    ├── data_loader.py  # DataLoader
    ├── preprocessor.py # Preprocessor
    ├── embedder.py     # Embedder
    ├── reranker.py     # Reranker (KeyBERT + MMR)
    └── evaluator.py    # Evaluator
```

## 실험 실행

```bash
# 전체 실험 (Exp 1~4)
python main.py --exp all

# 개별 실험
python main.py --exp 1      # 전처리 적용 vs 미적용
python main.py --exp 2      # SciBERT vs MiniLM vs MPNet 모델 비교
python main.py --exp 3      # Lambda sweep (λ = 0.0 ~ 1.0)
python main.py --exp 4      # 정성 분석

# 파라미터 오버라이드
python main.py --exp 3 --num_samples 50 --mmr_lambda 0.5
```

결과는 `results/<실험명>/`에 CSV, Markdown, 그래프로 저장됩니다.

## 실험 구성

| 실험 | 비교 변수 | 저장 경로 |
|---|---|---|
| Exp 1 | 전처리 적용 vs 미적용 | `results/exp1_preprocessing/` |
| Exp 2 | SciBERT vs MPNet vs MiniLM-L6 (KeyBERT 기본) | `results/exp2_encoder/` |
| Exp 3 | λ = [0.0, 0.1, ..., 1.0] sweep | `results/exp3_lambda_sweep/` |
| Exp 4 | 샘플별 정성 분석 | `results/exp4_qualitative/` |
