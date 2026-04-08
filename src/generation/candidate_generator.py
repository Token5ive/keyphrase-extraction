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
    model_path: str = "./outputs/scibart"
    input_col: str = "source_text"
    id_col: str = "id"

    max_new_tokens: int = 96

    num_beams: int = 20
    num_return_sequences: int = 10

    batch_size: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    output_json_path: str = "./results/predictions.json"
    output_csv_path: str = "./results/predictions.csv"


class CandidateGenerator:
    def __init__(self, config: Optional[CandidateGeneratorConfig] = None):
        self.config = config or CandidateGeneratorConfig()

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(self.config.device)
        self.model.eval()

        self.stemmer = PorterStemmer()

    def simple_tokenize(self, text: str):
        text = str(text).lower()
        return re.findall(r"[a-z0-9]+", text)

    def stem_text(self, text: str):
        return " ".join(self.stemmer.stem(t) for t in self.simple_tokenize(text))

    def parse_generated_text(self, generated_text: str) -> List[str]:
        text = str(generated_text).strip()

        # 1) 정상적인 semicolon 형식
        if ";" in text:
            candidates = [kp.strip() for kp in text.split(";") if kp.strip()]
        else:
            candidates = re.findall(r"'([^']+)'", text)

            # 혹시 작은따옴표가 없는 이상한 케이스 대비
            if not candidates:
                candidates = [text] if text else []

        # 후처리: 양끝 공백/중복 제거
        cleaned = []
        seen = set()

        for kp in candidates:
            kp = kp.strip()
            if not kp:
                continue

            norm = kp.lower()
            if norm not in seen:
                seen.add(norm)
                cleaned.append(kp)

        return cleaned

    def split_present_absent(self, source_text, kps):
        src = self.stem_text(source_text)

        present, absent = [], []
        for kp in kps:
            if self.stem_text(kp) in src:
                present.append(kp)
            else:
                absent.append(kp)

        return present, absent

    @torch.no_grad()
    def generate_batch(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.config.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            num_beams=self.config.num_beams,
            num_return_sequences=self.config.num_return_sequences,
            early_stopping=True,
        )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        results = []
        bs = len(texts)

        for i in range(bs):
            seqs = decoded[i * self.config.num_return_sequences:(i + 1) * self.config.num_return_sequences]

            all_kps = []
            for s in seqs:
                all_kps.extend(self.parse_generated_text(s))

            # deduplicate
            seen = set()
            unique = []
            for kp in all_kps:
                if kp.lower() not in seen:
                    seen.add(kp.lower())
                    unique.append(kp)

            results.append(unique)

        return results

    def generate(self, dataset):
        results = []

        for i in tqdm(range(0, len(dataset), self.config.batch_size)):
            batch = dataset[i:i + self.config.batch_size]
            texts = batch[self.config.input_col]

            preds = self.generate_batch(texts)

            for j in range(len(texts)):
                row = dataset[i + j]

                pred_kps = preds[j]
                present, absent = self.split_present_absent(texts[j], pred_kps)

                results.append({
                    "id": row[self.config.id_col],
                    "source_text": texts[j],
                    "predicted_kps": pred_kps,
                    "present_kps": present,
                    "absent_kps": absent,
                })

        return results

    def save(self, results):
        os.makedirs("./results", exist_ok=True)

        # JSON
        with open(self.config.output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # CSV
        df = pd.DataFrame(results)
        df["predicted_kps"] = df["predicted_kps"].apply(lambda x: " ; ".join(x))
        df["present_kps"] = df["present_kps"].apply(lambda x: " ; ".join(x))
        df["absent_kps"] = df["absent_kps"].apply(lambda x: " ; ".join(x))

        df.to_csv(self.config.output_csv_path, index=False)