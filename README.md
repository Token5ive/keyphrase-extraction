## Project Structure

```bash
keyphrase-extraction/
│
├─ sampled_01_preprocessed/
│   ├─ test/
│   │   ├─ data-00000-of-00001.arrow
│   │   ├─ dataset_info.json
│   │   └─ state.json
│   ├─ train/
│   │   ├─ data-00000-of-00001.arrow
│   │   ├─ dataset_info.json
│   │   └─ state.json
│   └─ validation/
│   │   ├─ data-00000-of-00001.arrow
│   │   ├─ dataset_info.json
│   │   └─ state.json
│
├─ src/
│   ├─ training/
│   │   ├─ __init__.py
│   │   ├─ kp_dataset.py
│   │   └─ scibart_trainer.py
│   │
│   ├─ generation/
│   │   ├─ __init__.py
│   │   └─ candidate_generator.py
│
├─ results/
│   ├─ predictions.json
│   └─ predictions.csv
│ 
├─ run_finetuning.py
├─ run_candidate_generation.py
├─ requirements.txt
└─ README.md

1 가상 환경 생성
```bash
python -m venv .venv
source .venv/bin/activate

2 라이브러리 설치
pip install -r requirements.txt

3 SciBART transformers 설치
pip uninstall -y transformers

git clone https://github.com/xiaowu0162/transformers.git -b scibart-integration
cd transformers
pip install -e .
cd ..


데이터셋
입력 데이터
	- HuggingFace Arrow format (.arrow)
	- 전처리 완료 데이터 사용
Model: uclanlp/scibart-large
Task: Seq2Seq Keyphrase Generation
Tokenizer: SciBART tokenizer


Training
python run_finetuning.py

수행 과정
1. .arrow 데이터 로드
2. 컬럼 검증
3. target_text 정규화 (list → string)
4. 데이터 유효성 검사
5. tokenizer 적용
6. SciBART fine-tuning
7. 모델 저장 (outputs/scibart/)


Generation
python run_candidate_generation.py

수행 과정
1. test 데이터 로드
2. 학습된 모델 불러오기
3. beam search 기반 keyphrase 생성
4. 중복 제거
5. present / absent 분리
6. 결과 저장