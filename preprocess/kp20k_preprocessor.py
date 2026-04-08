import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd
from nltk.stem import PorterStemmer


@dataclass
class KP20KPreprocessorConfig:
    sep_token: str = "[SEP]"
    kp_sep_token: str = " ; "
    lowercase: bool = True

    # Present / Absent 분리 기준
    use_prmu: bool = False

    # 디버깅/분석용 stemmed 컬럼 저장 여부
    save_stemmed_columns: bool = False

    # filtering 기준
    max_abstract_words: int = 512
    max_keyphrases: int = 15
    drop_duplicates: bool = True


class KP20KPreprocessor:
    def __init__(self, config: Optional[KP20KPreprocessorConfig] = None):
        self.config = config if config is not None else KP20KPreprocessorConfig()
        self.stemmer = PorterStemmer()

    # -----------------------------
    # basic text processing
    # -----------------------------
    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text).strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def normalize_text(self, text: str) -> str:
        text = self.clean_text(text)
        if self.config.lowercase:
            text = text.lower()
        return text

    def simple_tokenize(self, text: str) -> List[str]:
        text = self.normalize_text(text)
        return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text)

    # -----------------------------
    # stemming
    # -----------------------------
    def stem_phrase(self, phrase: str) -> str:
        tokens = self.simple_tokenize(phrase)
        return " ".join(self.stemmer.stem(tok) for tok in tokens)

    def stem_text(self, text: str) -> str:
        tokens = self.simple_tokenize(text)
        return " ".join(self.stemmer.stem(tok) for tok in tokens)

    # -----------------------------
    # build model input
    # -----------------------------
    def build_input_text(self, title: str, abstract: str) -> str:
        return f"{title} {self.config.sep_token} {abstract}"

    # -----------------------------
    # filtering helpers
    # -----------------------------
    def count_words(self, text: str) -> int:
        return len(self.simple_tokenize(text))

    def build_dedup_key(self, title: str, abstract: str) -> str:
        input_text = self.build_input_text(title, abstract)
        return self.normalize_text(input_text)

    # -----------------------------
    # present / absent split
    # -----------------------------
    def split_by_stemming(
        self,
        title: str,
        abstract: str,
        keyphrases: List[str]
    ) -> Tuple[List[str], List[str]]:
        source_text = self.build_input_text(title, abstract)
        stemmed_source_text = self.stem_text(source_text)

        present_kps = []
        absent_kps = []

        for kp in keyphrases:
            kp_clean = self.clean_text(kp)
            kp_stem = self.stem_phrase(kp_clean)

            if kp_stem and kp_stem in stemmed_source_text:
                present_kps.append(kp_clean)
            else:
                absent_kps.append(kp_clean)

        return present_kps, absent_kps

    def split_by_prmu(
        self,
        keyphrases: List[str],
        prmu: List[str]
    ) -> Tuple[List[str], List[str]]:
        present_kps = []
        absent_kps = []

        for kp, tag in zip(keyphrases, prmu):
            kp_clean = self.clean_text(kp)
            if tag == "P":
                present_kps.append(kp_clean)
            else:
                absent_kps.append(kp_clean)

        return present_kps, absent_kps

    # -----------------------------
    # target formatting
    # -----------------------------
    def join_keyphrases(self, keyphrases: List[str]) -> str:
        return self.config.kp_sep_token.join(keyphrases)

    # -----------------------------
    # dataframe filtering
    # -----------------------------
    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # text 정리
        df["title"] = df["title"].apply(self.clean_text)
        df["abstract"] = df["abstract"].apply(self.clean_text)

        # 1) abstract 길이 기준 필터링
        df["abstract_word_count"] = df["abstract"].apply(self.count_words)
        df = df[df["abstract_word_count"] <= self.config.max_abstract_words]

        # 2) keyphrase 개수 기준 필터링
        df["num_keyphrases_raw"] = df["keyphrases"].apply(len)
        df = df[df["num_keyphrases_raw"] <= self.config.max_keyphrases]

        # 3) 중복 문서 제거
        if self.config.drop_duplicates:
            df["dedup_key"] = df.apply(
                lambda row: self.build_dedup_key(row["title"], row["abstract"]),
                axis=1
            )
            df = df.drop_duplicates(subset=["dedup_key"], keep="first")

        df = df.reset_index(drop=True)
        return df

    # -----------------------------
    # one-row processing
    # -----------------------------
    def process_row(self, row: pd.Series) -> pd.Series:
        title = row["title"]
        abstract = row["abstract"]
        keyphrases = row["keyphrases"]

        input_text = self.build_input_text(title, abstract)

        if self.config.use_prmu and "prmu" in row.index:
            present_kps, absent_kps = self.split_by_prmu(keyphrases, row["prmu"])
        else:
            present_kps, absent_kps = self.split_by_stemming(title, abstract, keyphrases)

        row["input_text"] = input_text
        row["present_kps"] = present_kps
        row["absent_kps"] = absent_kps

        row["num_keyphrases"] = len(keyphrases)
        row["num_present_kps"] = len(present_kps)
        row["num_absent_kps"] = len(absent_kps)

        row["target_all_kps"] = self.join_keyphrases(keyphrases)
        row["target_present_kps"] = self.join_keyphrases(present_kps)
        row["target_absent_kps"] = self.join_keyphrases(absent_kps)

        if self.config.save_stemmed_columns:
            row["input_text_stemmed"] = self.stem_text(input_text)
            row["keyphrases_stemmed"] = [self.stem_phrase(kp) for kp in keyphrases]
            row["present_kps_stemmed"] = [self.stem_phrase(kp) for kp in present_kps]
            row["absent_kps_stemmed"] = [self.stem_phrase(kp) for kp in absent_kps]

        return row

    # -----------------------------
    # full preprocessing
    # -----------------------------
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.filter_dataframe(df)
        df = df.apply(self.process_row, axis=1)
        return df.reset_index(drop=True)

    def preprocess_splits(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame
    ):
        return (
            self.preprocess_dataframe(df_train),
            self.preprocess_dataframe(df_val),
            self.preprocess_dataframe(df_test),
        )