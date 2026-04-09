"""
DataLoader: predictions.json 로드 및 실험용 포맷 파싱.

predictions.json 구조:
  - id              : 샘플 식별자
  - source_text     : "generate keyphrases: title: <title> abstract: <abstract>"
  - predicted_kps   : SciBART가 생성한 후보 키프레이즈 리스트
  - pred_present_kps: 후보 중 문서 내 등장하는 키프레이즈
  - pred_absent_kps : 후보 중 문서에 등장하지 않는 키프레이즈
  - gold_target_kps : gold 키프레이즈 전체 리스트
  - gold_present_kps: 문서 내 등장하는 gold 키프레이즈
  - gold_absent_kps : 문서에 등장하지 않는 gold 키프레이즈
"""

import json
import re


def _parse_source_text(source_text: str) -> tuple[str, str]:
    """
    "generate keyphrases: title: <title> abstract: <abstract>" 형식에서
    title과 abstract를 분리하여 반환.
    """
    # "title: ... abstract: ..." 패턴 파싱
    m = re.search(r'title:\s*(.*?)\s+abstract:\s*(.*)', source_text, re.IGNORECASE | re.DOTALL)
    if m:
        title = m.group(1).strip()
        abstract = m.group(2).strip()
    else:
        # fallback: source_text 전체를 document로 사용
        title = ''
        abstract = source_text.strip()
    return title, abstract


class DataLoader:
    """predictions.json을 로드하고 실험에 필요한 레코드 리스트로 변환."""

    def __init__(self, data_path: str):
        self.data_path = data_path

    def load(self, num_samples: int = None) -> list[dict]:
        """
        JSON 파일을 읽어 레코드 리스트 반환.

        각 레코드:
          - id          : 샘플 ID
          - title       : 논문 제목
          - abstract    : 논문 초록
          - document    : title + ' . ' + abstract (임베딩 입력용)
          - candidates            : 후보 키프레이즈 리스트 (SciBART 예측, 생성 순서 유지)
          - candidate_positions   : 각 후보의 원본 생성 순서 인덱스 (beam score proxy)
          - n_candidates_original : 전처리 전 원본 후보 수 (beam score 정규화 기준)
          - keyphrases            : gold 키프레이즈 리스트 (present + absent)

        Args:
            num_samples: 로드할 최대 샘플 수. None이면 전체.
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        if num_samples is not None:
            raw = raw[:num_samples]

        records = []
        for item in raw:
            title, abstract = _parse_source_text(item['source_text'])
            candidates = list(item.get('predicted_kps', []))
            records.append({
                'id':                   item['id'],
                'title':                title,
                'abstract':             abstract,
                'document':             title + ' . ' + abstract,
                'candidates':           candidates,
                # 각 후보의 원본 생성 순서 (beam score proxy: 낮을수록 높은 신뢰도)
                'candidate_positions':  list(range(len(candidates))),
                # 전처리 후에도 정규화 기준으로 사용할 원본 후보 수
                'n_candidates_original': len(candidates),
                'keyphrases':           item.get('gold_present_kps', []) + item.get('gold_absent_kps', []),
            })

        return records
