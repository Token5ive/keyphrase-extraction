"""
Preprocessor: 후보 키프레이즈 전처리.

전처리 파이프라인:
  1. clean_candidate  — 소문자 변환, LaTeX/수식 제거, 특수기호 제거
  2. is_valid_candidate — 단어 수 · 알파벳 포함 여부 필터링
  3. deduplicate_candidates — 정규화 기준 중복 제거

use_preprocess=False 옵션으로 전처리를 생략하고 원본 후보 사용 가능.
"""

import re


class Preprocessor:
    """
    후보 키프레이즈 전처리기.

    Args:
        min_words: 유효 후보의 최소 단어 수 (기본 1)
        max_words: 유효 후보의 최대 단어 수 (기본 5)
    """

    def __init__(self, min_words: int = 1, max_words: int = 5):
        self.min_words = min_words
        self.max_words = max_words

    def clean(self, phrase: str) -> str:
        """
        단일 후보 키프레이즈 노이즈 제거.

        처리 순서:
            1. 소문자 변환
            2. 수식 · LaTeX 토큰 제거  (예: \\alpha, $x^2$, [MATH])
            3. 괄호 및 내용 제거       (예: (2019), [1])
            4. 특수기호 제거 (하이픈·슬래시는 유지)
            5. 연속 공백 정규화
        """
        phrase = phrase.lower().strip()

        # LaTeX / 수식 토큰 제거
        phrase = re.sub(r'\$[^$]*\$', ' ', phrase)
        phrase = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ' ', phrase)
        phrase = re.sub(r'\\[a-zA-Z]+', ' ', phrase)
        phrase = re.sub(r'\[MATH\]|\[FORMULA\]', ' ', phrase)

        # 괄호 및 내용 제거
        phrase = re.sub(r'\([^)]*\)', ' ', phrase)
        phrase = re.sub(r'\[[^\]]*\]', ' ', phrase)

        # 특수기호 제거 (하이픈·슬래시는 단어 구성 요소로 허용)
        phrase = re.sub(r'[^a-z0-9\s\-/]', ' ', phrase)

        # 연속 공백 정규화
        phrase = re.sub(r'\s+', ' ', phrase).strip()

        return phrase

    def is_valid(self, phrase: str) -> bool:
        """
        길이·내용 기준으로 유효한 후보인지 판별.

        기준:
            - 단어 수: min_words ≤ len ≤ max_words
            - 알파벳 문자가 1개 이상 포함
            - 숫자로만 구성된 토큰이 아님
        """
        words = phrase.split()
        if not (self.min_words <= len(words) <= self.max_words):
            return False
        if not re.search(r'[a-z]', phrase):
            return False
        if all(w.isdigit() for w in words):
            return False
        return True

    def deduplicate(self, candidates: list[str]) -> list[str]:
        """
        정규화 후 동일 표현인 후보 제거 (순서 유지, 첫 등장 우선).

        정규화 기준:
            - 소문자
            - 하이픈·슬래시를 공백으로 통일
            - 연속 공백 제거
        """
        seen = set()
        result = []
        for phrase in candidates:
            key = re.sub(r'[-/]', ' ', phrase)
            key = re.sub(r'\s+', ' ', key).strip()
            if key not in seen:
                seen.add(key)
                result.append(phrase)
        return result

    def process(self, candidates: list[str]) -> list[str]:
        """
        전처리 파이프라인 전체 실행.

        단계: clean → filter → deduplicate
        """
        cleaned = [self.clean(c) for c in candidates]
        filtered = [c for c in cleaned if self.is_valid(c)]
        return self.deduplicate(filtered)

    def apply_to_records(self, records: list[dict], use_preprocess: bool = True) -> list[dict]:
        """
        레코드 리스트의 candidates 필드에 전처리 적용.

        Args:
            records: DataLoader.load()가 반환한 레코드 리스트
            use_preprocess: True면 전처리 적용, False면 원본 유지

        Returns:
            candidates가 업데이트된 새 레코드 리스트 (원본 불변).
        """
        result = []
        for rec in records:
            new_rec = dict(rec)
            if use_preprocess:
                new_rec['candidates'] = self.process(rec['candidates'])
            result.append(new_rec)
        return result
