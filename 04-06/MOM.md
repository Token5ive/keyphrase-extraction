# 회의록(MOM)
## 1. 논의 사항 및 결과
1. 논의 사항
- 조사한 선행연구 바탕으로 데이터 전처리 및 모델링 설계

2. 논의 결과
- 데이터 전처리
  1) KP20k 데이터셋 (2.19.0 ver)
  2) 구조 확인, 결측치 확인, 길이 및 keyphrase 개수 분포 확인, PRMU 비율 확인, 시각화 및 수치 정리 
- 모델링
  1) backbone ; BART-based
  2) Present ; decoder copy mechanism
  3) Absent ; decoder seq2seq
  4) Ranker ; Contrastive reranker, MMR 후처리
- 역할 분담
  1) 데이터 ; 신준수, 손중영, 장동욱
  2) 모델링 ; 심소민, 이지선
