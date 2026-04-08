import json
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.stem import PorterStemmer


@dataclass
class CandidateGeneratorConfig:
    model_path: str = "outputs/scibart_ckpt_toy" # 실제로는 "outputs/scibart_ckpt" 이어야 함
    input_col: str = "input_text"
    id_col: str = "id"

    max_source_length: int = 512
    max_new_tokens: int = 96

    # 요청값을 그대로 넣을 수 있게 유지
    num_beams: int = 50
    num_return_sequences: int = 20

    batch_size: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    output_jsonl_path: str = "data/candidates/kp20k_toy_candidates.jsonl"
    output_csv_path: str = "data/candidates/kp20k_toy_candidates.csv"


class CandidateGenerator:
    def __init__(self, config: Optional[CandidateGeneratorConfig] = None):
        self.config = config if config is not None else CandidateGeneratorConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(self.config.device)
        self.model.eval()
        self.stemmer = PorterStemmer()

    def simple_tokenize(self, text: str) -> List[str]:
        text = str(text).strip().lower()
        return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text)

    def stem_text(self, text: str) -> str:
        return " ".join(self.stemmer.stem(tok) for tok in self.simple_tokenize(text))

    def parse_generated_text(self, generated_text: str) -> List[str]:
        candidates = [kp.strip() for kp in generated_text.split(";")]
        candidates = [kp for kp in candidates if kp]

        # 중복 제거 (순서 유지)
        seen = set()
        unique_candidates = []
        for kp in candidates:
            norm = kp.lower().strip()
            if norm not in seen:
                seen.add(norm)
                unique_candidates.append(kp)
        return unique_candidates

    def split_present_absent(self, source_text: str, keyphrases: List[str]):
        stemmed_source = self.stem_text(source_text)
        present, absent = [], []

        for kp in keyphrases:
            kp_stem = self.stem_text(kp)
            if kp_stem and kp_stem in stemmed_source:
                present.append(kp)
            else:
                absent.append(kp)

        return present, absent

    @torch.no_grad()
    def generate_for_batch(self, texts: List[str]) -> List[Dict]:
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_source_length,
            return_tensors="pt",
        ).to(self.config.device)

        outputs = self.model.generate(
            **encodings,
            max_new_tokens=self.config.max_new_tokens,
            num_beams=self.config.num_beams,
            num_return_sequences=self.config.num_return_sequences,
            early_stopping=True,
            no_repeat_ngram_size=2,
            return_dict_in_generate=True,
            output_scores=False,
        )

        decoded_sequences = self.tokenizer.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
        )

        batch_size = len(texts)
        grouped_results = []

        for i in range(batch_size):
            start = i * self.config.num_return_sequences
            end = (i + 1) * self.config.num_return_sequences
            group_seqs = decoded_sequences[start:end]

            all_candidates = []
            for seq in group_seqs:
                parsed = self.parse_generated_text(seq)
                all_candidates.extend(parsed)

            # 전체 beam 후보들 합친 뒤 중복 제거
            seen = set()
            unique_candidates = []
            for kp in all_candidates:
                norm = kp.lower().strip()
                if norm not in seen:
                    seen.add(norm)
                    unique_candidates.append(kp)

            grouped_results.append({
                "generated_sequences": group_seqs,
                "candidate_keyphrases": unique_candidates,
            })

        return grouped_results

    def generate_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []

        rows = df.to_dict("records")

        for start_idx in tqdm(range(0, len(rows), self.config.batch_size), desc="Generating candidates"):
            batch_rows = rows[start_idx:start_idx + self.config.batch_size]
            batch_texts = [row[self.config.input_col] for row in batch_rows]

            batch_outputs = self.generate_for_batch(batch_texts)

            for row, out in zip(batch_rows, batch_outputs):
                source_text = row[self.config.input_col]
                candidates = out["candidate_keyphrases"]
                present_candidates, absent_candidates = self.split_present_absent(source_text, candidates)

                results.append({
                    "id": row.get(self.config.id_col, None),
                    "input_text": source_text,
                    "generated_sequences": out["generated_sequences"],
                    "candidate_keyphrases": candidates,
                    "present_candidates": present_candidates,
                    "absent_candidates": absent_candidates,
                    "num_candidates": len(candidates),
                    "num_present_candidates": len(present_candidates),
                    "num_absent_candidates": len(absent_candidates),
                })

        return pd.DataFrame(results)

    def save_outputs(self, df_candidates: pd.DataFrame):
        os.makedirs(os.path.dirname(self.config.output_jsonl_path), exist_ok=True)

        df_candidates.to_csv(self.config.output_csv_path, index=False)

        with open(self.config.output_jsonl_path, "w", encoding="utf-8") as f:
            for row in df_candidates.to_dict("records"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")