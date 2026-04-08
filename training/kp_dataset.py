from dataclasses import dataclass
from typing import Optional

from datasets import Dataset


@dataclass
class Seq2SeqDatasetConfig:
    input_col: str = "input_text"
    target_col: str = "target_all_kps"
    max_source_length: int = 512
    max_target_length: int = 128


class KeyphraseSeq2SeqDatasetBuilder:
    def __init__(self, tokenizer, config: Optional[Seq2SeqDatasetConfig] = None):
        self.tokenizer = tokenizer
        self.config = config if config is not None else Seq2SeqDatasetConfig()

    def _tokenize_function(self, examples):
        model_inputs = self.tokenizer(
            examples[self.config.input_col],
            max_length=self.config.max_source_length,
            truncation=True,
            padding=False,
        )

        labels = self.tokenizer(
            text_target=examples[self.config.target_col],
            max_length=self.config.max_target_length,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def build(self, df):
        hf_dataset = Dataset.from_pandas(df, preserve_index=False)
        tokenized_dataset = hf_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=hf_dataset.column_names,
        )
        return tokenized_dataset