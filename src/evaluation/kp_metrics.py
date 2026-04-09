import json
import re
from dataclasses import dataclass
from typing import List, Dict

from nltk.stem import PorterStemmer


@dataclass
class EvalResult:
    precision: float
    recall: float
    f1: float


class KeyphraseEvaluator:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def simple_tokenize(self, text: str):
        text = str(text).lower()
        return re.findall(r"[a-z0-9]+", text)

    def normalize_phrase(self, phrase: str) -> str:
        tokens = self.simple_tokenize(phrase)
        stems = [self.stemmer.stem(t) for t in tokens]
        return " ".join(stems)

    def normalize_list(self, phrases: List[str]) -> List[str]:
        normalized = []
        seen = set()

        for phrase in phrases:
            norm = self.normalize_phrase(phrase)
            if norm and norm not in seen:
                seen.add(norm)
                normalized.append(norm)

        return normalized

    def compute_example_scores(self, pred: List[str], gold: List[str]) -> Dict[str, float]:
        pred_norm = set(self.normalize_list(pred))
        gold_norm = set(self.normalize_list(gold))

        if len(pred_norm) == 0 and len(gold_norm) == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        if len(pred_norm) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        correct = len(pred_norm & gold_norm)

        precision = correct / len(pred_norm) if pred_norm else 0.0
        recall = correct / len(gold_norm) if gold_norm else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def compute_macro_scores(self, results: List[Dict], pred_key: str, gold_key: str) -> EvalResult:
        precisions, recalls, f1s = [], [], []

        for row in results:
            scores = self.compute_example_scores(row[pred_key], row[gold_key])
            precisions.append(scores["precision"])
            recalls.append(scores["recall"])
            f1s.append(scores["f1"])

        n = len(results)
        return EvalResult(
            precision=sum(precisions) / n,
            recall=sum(recalls) / n,
            f1=sum(f1s) / n,
        )

    def load_json(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)