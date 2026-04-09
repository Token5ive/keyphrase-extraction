"""
DataLoader: predictions.json 로드 및 실험용 포맷 파싱.

predictions.json 구조:
  - id          : 샘플 식별자
  - source_text : "generate keyphrases: title: <title> abstract: <abstract>"
  - predicted_kps : SciBART가 생성한 후보 키프레이즈 리스트
  - present_kps   : 문서 내 등장하는 gold 키프레이즈
  - absent_kps    : 문서에 등장하지 않는 gold 키프레이즈
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
          - candidates  : 후보 키프레이즈 리스트 (SciBART 예측)
          - keyphrases  : gold 키프레이즈 리스트 (present + absent)

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
            records.append({
                'id':         item['id'],
                'title':      title,
                'abstract':   abstract,
                'document':   title + ' . ' + abstract,
                'candidates': list(item.get('predicted_kps', [])),
                'keyphrases': item.get('present_kps', []) + item.get('absent_kps', []),
            })

        return records
