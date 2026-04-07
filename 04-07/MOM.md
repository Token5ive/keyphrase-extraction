# 회의록(MOM)
## 논의 사항 및 결과
1. 논의 사항
- 데이터 EDA, 전처리 방법 논의
- 모델 소스코드 리뷰 (현재 진행 상황까지)

2. 논의 결과
- 데이터 전처리
  1) KP20k 데이터셋 (2.19.0 ver)
  2) 전처리 파일 형식 ; .arrow 확장자
  3) 토큰화 ; uclanlp/scibart-large 사용
  4) PRMU ; P(extract), RMU(absent)
  5) 중복 문서, 결측치 행 전부 제거
  6) 전처리 파일 컬럼 통일
- 모델링
  1) keybart 대신 scibart 사용
  2) 후보 생성(generator) 시, 결과 파일은 .json 확장자로 전달
  3) .arrow에 맞춰서 데이터셋 변환 / 토큰화 파트 제거
