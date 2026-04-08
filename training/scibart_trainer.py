from dataclasses import dataclass
from typing import Optional

from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

from .kp_dataset import KeyphraseSeq2SeqDatasetBuilder, Seq2SeqDatasetConfig


@dataclass
class SciBARTTrainerConfig:
    model_name: str = "uclanlp/scibart-large"
    output_dir: str = "./outputs/scibart"

    input_col: str = "source_text"
    target_col: str = "target_text"
    abstract_col: str = "abstract"

    batch_size: int = 4
    lr: float = 5e-5
    epochs: int = 3

    fp16: bool = False


class SciBARTFineTuner:
    def __init__(self, config: Optional[SciBARTTrainerConfig] = None):
        self.config = config or SciBARTTrainerConfig()

        print("Loading SciBART...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.config.model_name)

        self.builder = KeyphraseSeq2SeqDatasetBuilder(
            tokenizer=self.tokenizer,
            config=Seq2SeqDatasetConfig(
                input_col=self.config.input_col,
                target_col=self.config.target_col,
                abstract_col=self.config.abstract_col,
            ),
        )

        self.collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
        )

    def train(self, train_ds, val_ds):
        train_ds = self.builder.build(train_ds)
        val_ds = self.builder.build(val_ds)

        args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.lr,
            num_train_epochs=self.config.epochs,

            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",

            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",

            fp16=self.config.fp16,
            save_total_limit=2,
            report_to="none",
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=self.tokenizer,
            data_collator=self.collator,
            callbacks=[EarlyStoppingCallback(2)],
        )

        trainer.train()

        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

        return trainer