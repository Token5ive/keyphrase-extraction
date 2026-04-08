# KP20k/Inspec 데이터 전처리 모듈

이 폴더는 KP20k 및 Inspec 데이터셋 전처리를 담당합니다.

## 📚 지원 데이터셋

| 데이터셋 | 용도 | Split | HuggingFace |
|---------|------|-------|-------------|
| **KP20k** | 학습/검증/평가 | train, validation, test | `taln-ls2n/kp20k` |
| **Inspec** | 평가 전용 | test | `midas/inspec` |

## 📦 필수 라이브러리 설치

**중요**: `datasets` 라이브러리 버전이 매우 중요합니다!

```bash
pip install datasets==2.19.0 pandas numpy pyarrow==15.0.0
```

### 라이브러리 버전 명세

```
datasets==2.19.0       # 필수! kp20k 데이터셋 호환 버전
pandas>=2.0.0
numpy>=1.20.0
pyarrow==15.0.0        # datasets 2.19.0과 호환
```

### ⚠️ 버전 관련 주의사항

- **datasets 2.14.0 이하**: `LocalFileSystem` 캐싱 버그로 실행 불가
- **datasets 4.x 이상**: dataset script 완전 차단으로 kp20k 로드 불가
- **datasets 2.19.0**: kp20k와 완벽하게 호환되는 안정 버전 ✅
- **pyarrow 23.0 이상**: datasets 2.19.0과 호환 문제 발생

Python 3.12 환경에서 테스트 완료.

---

## 🚀 사용 방법

### 기본 사용법 (KP20k)

```python
from preprocessing import run_pipeline

# KP20k 전처리 실행 및 저장
results = run_pipeline(dataset="kp20k", save=True, base_save_path="output")

# 각 split별 DataFrame 접근
train_df = results['train']
val_df = results['validation']
test_df = results['test']
```

### Inspec 데이터셋 (평가용)

```python
from preprocessing import run_pipeline

# Inspec 전처리 (test split만 존재)
results = run_pipeline(dataset="inspec", save=True, base_save_path="output")

# test split 접근
test_df = results['test']
```

### 저장 없이 DataFrame만 반환

```python
from preprocessing import run_pipeline

# 메모리에만 로드
kp20k_results = run_pipeline(dataset="kp20k", save=False)
inspec_results = run_pipeline(dataset="inspec", save=False)
```

### 저장 경로 커스터마이징

```python
from preprocessing import run_pipeline

# custom_output 폴더에 저장
results = run_pipeline(dataset="kp20k", save=True, base_save_path="custom_output")
```

---

## 📊 출력 데이터 구조

### 저장 위치

**KP20k 기본 저장 구조**
```
output/
├── kp20k_train_clean/         # Arrow 포맷 데이터셋
├── kp20k_validation_clean/
└── kp20k_test_clean/
```

**Inspec 기본 저장 구조**
```
output/
└── inspec_test_clean/         # Inspec은 test만 존재
```

### DataFrame 컬럼

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| `id` | 문서 고유 ID | "2365347" |
| `title` | 논문 제목 | "Deep Learning for NLP" |
| `abstract` | 초록 텍스트 | "This paper presents..." |
| `keyphrases` | 전체 키워드 리스트 | ["deep learning", "nlp"] |
| `prmu` | Present/Absent 태그 | ["P", "A"] |
| `present_kps` | 초록에 등장하는 키워드 | ["deep learning"] |
| `absent_kps` | 초록에 없는 키워드 | ["nlp"] |
| `source_text` | 모델 입력용 텍스트 | "generate keyphrases: title: ... abstract: ..." |
| `target_text` | 모델 타겟용 텍스트 | "['deep learning' 'nlp']" |

### 예상 데이터 크기

- **Train**: ~498,000개 문서
- **Validation**: ~18,000개 문서
- **Test**: ~18,000개 문서

---

## 🔧 전처리 세부 사항

### 자동 임계값 계산

각 split(train/validation/test)마다 **독립적으로** 임계값을 계산합니다.

#### 연속형 변수 (IQR 방식)
- **Abstract 길이**: Q3 + 1.5 × IQR (Tukey's rule)
- **KP 문자 수**: Q3 + 1.5 × IQR

#### 이산형 변수 (Percentile 방식)
- **KP 개수**: 95th percentile
- **KP 단어 수**: 95th percentile

### 전처리 단계 순서

1. 초록 없는 문서 제거
2. 긴 문서 제거 (IQR 기준)
3. 중복 문서 제거
4. KP 많은 문서 제거 (95th percentile)
5. 긴 단어 KP 제거 (95th percentile)
6. 긴 문자 KP 제거 (IQR 기준)

---

## 💡 개별 함수 사용법

### 1. 데이터 로드만 하기

**KP20k 로드**
```python
from preprocessing import load_kp20k

splits = load_kp20k()
train_df = splits['train']
val_df = splits['validation']
test_df = splits['test']
```

**Inspec 로드**
```python
from preprocessing import load_inspec

splits = load_inspec()
test_df = splits['test']  # Inspec은 test만 존재
```

**통합 로더 사용**
```python
from preprocessing import load_dataset_by_name

# KP20k 또는 Inspec 자동 로드
kp20k_splits = load_dataset_by_name("kp20k")
inspec_splits = load_dataset_by_name("inspec")
```

### 2. 임계값만 계산하기

```python
from preprocessing.filter_data import compute_thresholds
import pandas as pd

df = pd.DataFrame(...)  # 사용자 데이터
thresholds = compute_thresholds(df)

print(thresholds)
# {'max_words': 316, 'max_kp': 9, 'max_kp_words': 4, 'max_kp_chars': 37}
```

### 3. 전처리 단계만 실행

```python
from preprocessing.filter_data import run_preprocessing
import pandas as pd

df = pd.DataFrame(...)  # 원본 데이터
cleaned_df = run_preprocessing(df, split="train")
```

### 4. 컬럼 생성만 하기

```python
from preprocessing.build_column import build_columns

df_with_columns = build_columns(cleaned_df)
```

### 5. Arrow 포맷으로 저장

```python
from preprocessing.build_column import save_to_arrow

save_to_arrow(df_final, "output/my_custom_clean")
```

---

## 📝 통합 예시 코드

```python
# 모델링 파일에서 사용하는 예시
import sys
sys.path.append('.')  # preprocessing 폴더가 있는 경로

from preprocessing import run_pipeline

def main():
    # 1. KP20k 학습용 데이터 전처리
    print("=== KP20k 전처리 시작 ===")
    kp20k_results = run_pipeline(
        dataset="kp20k",
        save=True,
        base_save_path="data/processed"
    )
    
    for split_name, df in kp20k_results.items():
        print(f"{split_name}: {len(df):,}개 문서")
    
    # 2. Inspec 평가용 데이터 전처리
    print("\n=== Inspec 전처리 시작 ===")
    inspec_results = run_pipeline(
        dataset="inspec",
        save=True,
        base_save_path="data/processed"
    )
    
    print(f"test: {len(inspec_results['test']):,}개 문서")
    
    print("\n=== 전처리 완료 ===")
    print("KP20k 저장 위치: data/processed/kp20k_{train|validation|test}_clean/")
    print("Inspec 저장 위치: data/processed/inspec_test_clean/")
    
    return kp20k_results, inspec_results

if __name__ == "__main__":
    kp20k_data, inspec_data = main()
```

---

## 🔍 데이터셋 통계 (참고)

### KP20k 데이터셋

**Train Split**
- 원본: 530,809개 → 전처리 후: ~498,000개 (약 94% 유지)
- 자동 기준 예시: abstract≤316단어, kp≤9개, kp단어≤4, kp문자≤37자

**Validation Split**
- 원본: 20,000개 → 전처리 후: ~18,500개 (약 92% 유지)
- 자동 기준 예시: abstract≤314단어, kp≤9개, kp단어≤4, kp문자≤37자

**Test Split**
- 원본: 20,000개 → 전처리 후: ~18,500개 (약 92% 유지)
- 자동 기준 예시: abstract≤318단어, kp≤9개, kp단어≤4, kp문자≤37자

### Inspec 데이터셋

**Test Split**
- 원본: ~500개 → 전처리 후: 데이터 분포에 따라 자동 결정
- 평가 전용으로 사용
- KP20k와 독립적인 임계값 자동 계산

---

## 📌 주요 특징

✅ **멀티 데이터셋 지원**: KP20k (학습용) + Inspec (평가용) 통합 처리  
✅ **자동 임계값 계산**: 데이터셋별로 통계 기반 필터링 기준 자동 설정  
✅ **Split별 독립 처리**: train/val/test 각각 다른 분포에 맞게 처리  
✅ **통계적 근거**: IQR (연속형), Percentile (이산형) 방식 적용  
✅ **성능 최적화**: explode() 활용으로 10배 빠른 처리  
✅ **엣지 케이스 대응**: 빈 데이터, null 값 안전 처리  

---
