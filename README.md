# KP20k Keyphrase Generation Project

KP20k 데이터셋을 사용해 전처리 → SciBART 파인튜닝 → 후보 생성까지 수행하는 프로젝트입니다.

## Project Structure

```bash
project/
│
├─ data/
│   ├─ processed/
│   └─ candidates/
│
├─ outputs/
│   └─ scibart_ckpt/
│
├─ src/
│   ├─ preprocess/
│   │   ├─ __init__.py
│   │   └─ kp20k_preprocessor.py
│   │
│   ├─ training/
│   │   ├─ __init__.py
│   │   ├─ kp_dataset.py
│   │   └─ scibart_trainer.py
│   │
│   ├─ generation/
│   │   ├─ __init__.py
│   │   └─ candidate_generator.py
│
├─ run_preprocessing.py
├─ run_finetuning.py
├─ run_candidate_generation.py
├─ requirements.txt
├─ setup_scibart.sh
└─ README.md

방법 1 (권장) SciBART 환경 자동 세팅
```bash
bash setup_scibart.sh
source .venv/bin/activate

방법 2 (수동 설치)
```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch pandas datasets==2.19.0 tqdm nltk accelerate sentencepiece

git clone https://github.com/xiaowu0162/transformers.git -b scibart-integration
cd transformers
pip install -e .
cd ..


Preprocessing Rules
	•	입력 형식: title [SEP] abstract
	•	필터링 기준: 
	    •	abstract 단어 수 > 512 → 제거
	    •	keyphrase 개수 > 15 → 제거
	    •	중복 문서 제거 (title + abstract 기준)
	•	keyphrase 처리:
	    •	Porter Stemmer 적용
	    •	Present / Absent 분리
	•	출력 포맷: kp1 ; kp2 ; kp3



# Summary
이 프로젝트는 다음 파이프라인으로 구성됩니다:

Preprocessing
   ↓
SciBART Fine-tuning
   ↓
Candidate Generation
   ↓
(향후) Reranker