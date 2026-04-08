from dataclasses import dataclass
from typing import Optional

from datasets import Dataset


@dataclass
class Seq2SeqDatasetConfig:
    input_col: str = "source_text"
    target_col: str = "target_text"
    abstract_col: str = "abstract"


class KeyphraseSeq2SeqDatasetBuilder:
    def __init__(self, tokenizer, config: Optional[Seq2SeqDatasetConfig] = None):
        self.tokenizer = tokenizer
        self.config = config if config is not None else Seq2SeqDatasetConfig()

    def _validate_required_columns(self, dataset: Dataset):
        required_cols = {self.config.input_col, self.config.target_col, self.config.abstract_col}
        missing = required_cols - set(dataset.column_names)

        if missing:
            raise ValueError(f"Missing columns: {missing}")

    @staticmethod
    def _is_nonempty_text(x):
        return isinstance(x, str) and x.strip() != ""

    def _normalize_target_text(self, example):
        value = example[self.config.target_col]

        if isinstance(value, str):
            return example

        try:
            example[self.config.target_col] = " ; ".join(map(str, list(value)))
            return example
        except:
            raise ValueError(f"Invalid target_text format: {type(value)}")

    def _validate_dataset_strict(self, dataset: Dataset):
        invalid = []

        for i in range(len(dataset)):
            ex = dataset[i]

            if not self._is_nonempty_text(ex.get(self.config.abstract_col)):
                invalid.append((i, "abstract"))

            elif not self._is_nonempty_text(ex.get(self.config.input_col)):
                invalid.append((i, "source_text"))

            elif not self._is_nonempty_text(ex.get(self.config.target_col)):
                invalid.append((i, "target_text"))

        if invalid:
            raise ValueError(
                f"Invalid samples detected!\n"
                f"count={len(invalid)}\n"
                f"example={invalid[:5]}"
            )

        print(f"[Validation] OK (size={len(dataset)})")

    def _tokenize(self, examples):
        inputs = self.tokenizer(
            examples[self.config.input_col],
            truncation=False,
            padding=False,
        )

        labels = self.tokenizer(
            text_target=examples[self.config.target_col],
            truncation=False,
            padding=False,
        )

        inputs["labels"] = labels["input_ids"]
        return inputs

    def build(self, dataset: Dataset):
        self._validate_required_columns(dataset)

        dataset = dataset.map(self._normalize_target_text)
        self._validate_dataset_strict(dataset)

        dataset = dataset.map(
            self._tokenize,
            batched=True,
            remove_columns=dataset.column_names,
        )

        return dataset