import numpy as np
import pandas as pd


# ── 임계값 자동 계산 ─────────────────────────────────────────────────────────────
def compute_thresholds(df):
    """
    각 split별 임계값 자동 계산 (하드코딩 없음)
      - 연속형 feature: IQR 이상치 기준 (Tukey 1.5 IQR rule)
      - 이산형 feature: 95th percentile (이산 분포 특성 반영)
      - 엣지 케이스: 빈 데이터 시 기본값 반환
    """
    # 빈 DataFrame 예외 처리
    if len(df) == 0:
        return {
            "max_words": 300,
            "max_kp": 10,
            "max_kp_words": 5,
            "max_kp_chars": 50,
        }

    # Abstract 길이: 연속형 → IQR
    abstract_lens = df["abstract"].str.split().str.len().dropna()
    if len(abstract_lens) > 0:
        Q1, Q3 = np.percentile(abstract_lens, [25, 75])
        max_words = int(Q3 + 1.5 * (Q3 - Q1))
    else:
        max_words = 300

    # KP 개수: 이산형 → 95th percentile
    kp_counts = df["keyphrases"].apply(len)
    max_kp = int(np.percentile(kp_counts, 95)) if len(kp_counts) > 0 else 10

    # KP 단어수: 이산형 → 95th percentile (explode로 성능 개선)
    kp_word_lens = df["keyphrases"].explode().apply(lambda kp: len(kp.split()) if pd.notna(kp) else 0)
    max_kp_words = int(np.percentile(kp_word_lens, 95)) if len(kp_word_lens) > 0 else 5

    # KP 문자수: 연속형 → IQR (explode로 성능 개선)
    kp_char_lens = df["keyphrases"].explode().apply(lambda kp: len(kp) if pd.notna(kp) else 0)
    if len(kp_char_lens) > 0:
        Q1, Q3 = np.percentile(kp_char_lens, [25, 75])
        max_kp_chars = int(Q3 + 1.5 * (Q3 - Q1))
    else:
        max_kp_chars = 50

    return {
        "max_words":    max_words,
        "max_kp":       max_kp,
        "max_kp_words": max_kp_words,
        "max_kp_chars": max_kp_chars,
    }


# ── 개별 전처리 함수들 ───────────────────────────────────────────────────────────
def remove_no_abstract(df):
    """초록 없는 문서 제거"""
    return df[df["abstract"].notna() & (df["abstract"].str.strip() != "")]


def remove_long_docs(df, max_words):
    """긴 문서 제거 (IQR 기준)"""
    return df[df["abstract"].str.split().str.len() <= max_words]


def remove_many_kp(df, max_kp):
    """KP 개수 초과 문서 제거 (95th percentile 기준)"""
    return df[df["keyphrases"].apply(len) <= max_kp]


def remove_duplicate_docs(df):
    """중복 문서 제거"""
    return df.drop_duplicates(subset=["abstract"]).reset_index(drop=True)


def remove_wordy_kp(df, max_words):
    """단어 수 많은 KP 제거 (95th percentile 기준)"""
    df = df.copy()
    before = df["keyphrases"].apply(len).sum()

    def filter_row(row):
        filtered = [
            (kp, p)
            for kp, p in zip(row["keyphrases"], row["prmu"])
            if len(kp.split()) <= max_words
        ]
        if not filtered:
            return None
        row["keyphrases"] = [kp for kp, _ in filtered]
        row["prmu"]       = [p  for _,  p in filtered]
        return row

    df = df.apply(filter_row, axis=1).dropna().reset_index(drop=True)
    print(f"    → 제거된 KP 개수: {before - df['keyphrases'].apply(len).sum()}개")
    return df


def remove_long_kp(df, max_chars):
    """문자 수 많은 KP 제거 (IQR 기준)"""
    df = df.copy()
    before = df["keyphrases"].apply(len).sum()

    def filter_row(row):
        filtered = [
            (kp, p)
            for kp, p in zip(row["keyphrases"], row["prmu"])
            if len(kp) <= max_chars
        ]
        if not filtered:
            return None
        row["keyphrases"] = [kp for kp, _ in filtered]
        row["prmu"]       = [p  for _,  p in filtered]
        return row

    df = df.apply(filter_row, axis=1).dropna().reset_index(drop=True)
    print(f"    → 제거된 KP 개수: {before - df['keyphrases'].apply(len).sum()}개")
    return df


# ── 메인 파이프라인 ──────────────────────────────────────────────────────────────
def run_preprocessing(df, split="train"):
    """
    각 split마다 임계값을 자동 계산하여 전처리 실행

    적용 기준:
      Abstract 길이  : IQR 이상치 기준 (Tukey 1.5 IQR rule, 연속형)
      KP 개수        : 95th percentile (이산형)
      KP 단어수      : 95th percentile (이산형)
      KP 문자수      : IQR 이상치 기준 (연속형)
    """
    print(f"\n[{split}] 시작: {len(df):,}개")

    t = compute_thresholds(df)
    print(f"  [자동 기준] abstract≤{t['max_words']}단어 | "
          f"kp≤{t['max_kp']}개 | "
          f"kp단어≤{t['max_kp_words']} | "
          f"kp문자≤{t['max_kp_chars']}자")

    df = remove_no_abstract(df)
    print(f"  초록 없는 문서 제거 후  : {len(df):,}개")

    df = remove_long_docs(df, t["max_words"])
    print(f"  긴 문서 제거 후         : {len(df):,}개")

    df = remove_duplicate_docs(df)
    print(f"  중복 문서 제거 후       : {len(df):,}개")

    df = remove_many_kp(df, t["max_kp"])
    print(f"  KP 많은 문서 제거 후    : {len(df):,}개")

    df = remove_wordy_kp(df, t["max_kp_words"])
    print(f"  긴 단어 KP 제거 후      : {len(df):,}개")

    df = remove_long_kp(df, t["max_kp_chars"])
    print(f"  긴 문자 KP 제거 후      : {len(df):,}개")

    return df