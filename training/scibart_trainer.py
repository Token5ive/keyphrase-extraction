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
    output_dir: str = "outputs/scibart_ckpt"
    input_col: str = "input_text"
    target_col: str = "target_all_kps"

    max_source_length: int = 512
    max_target_length: int = 128 

    per_device_train_batch_size: int = 4    
    per_device_eval_batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 10

    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_strategy: str = "epoch"

    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    early_stopping_patience: int = 2 
    early_stopping_threshold: float = 0.0 

    fp16: bool = False
    predict_with_generate: bool = False
    save_total_limit: int = 2
    report_to: str = "none"


class SciBARTFineTuner:
    def __init__(self, config: Optional[SciBARTTrainerConfig] = None):
        self.config = config if config is not None else SciBARTTrainerConfig()

        # SciBART는 scibart-integration transformers 환경에서 로드해야 함
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.config.model_name)

        self.dataset_builder = KeyphraseSeq2SeqDatasetBuilder(
            tokenizer=self.tokenizer,
            config=Seq2SeqDatasetConfig(
                input_col=self.config.input_col,
                target_col=self.config.target_col,
                max_source_length=self.config.max_source_length,
                max_target_length=self.config.max_target_length,
            ),
        )

        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
        )

    def build_datasets(self, df_train, df_val):
        train_dataset = self.dataset_builder.build(df_train)
        val_dataset = self.dataset_builder.build(df_val)
        return train_dataset, val_dataset

    def get_training_args(self):
        return Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            evaluation_strategy=self.config.eval_strategy,
            save_strategy=self.config.save_strategy,
            logging_strategy=self.config.logging_strategy,

            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,

            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            num_train_epochs=self.config.num_train_epochs,

            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,

            predict_with_generate=self.config.predict_with_generate,
            fp16=self.config.fp16,
            save_total_limit=self.config.save_total_limit,
            report_to=self.config.report_to,
        )

    def train(self, df_train, df_val):
        train_dataset, val_dataset = self.build_datasets(df_train, df_val)

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.get_training_args(),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold,
                )
            ],
        )

        trainer.train()
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

        return trainer