import os
import pandas as pd

from src.training import SciBARTFineTuner, SciBARTTrainerConfig

def main():
    df_train = pd.read_pickle("data/processed/kp20k_train_toy_preprocessed.pkl")
    df_val = pd.read_pickle("data/processed/kp20k_val_toy_preprocessed.pkl")
    
    os.makedirs("outputs/scibart_ckpt_toy", exist_ok=True)

    trainer = SciBARTFineTuner(
        SciBARTTrainerConfig(
            model_name="uclanlp/scibart-large",
            output_dir="outputs/scibart_ckpt_toy",
            input_col="input_text",
            target_col="target_all_kps",
            max_source_length=512,
            max_target_length=128,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=5e-5,
            num_train_epochs=3,
            early_stopping_patience=2,
            early_stopping_threshold=0.0,
            fp16=False,
        )
    )

    trainer.train(df_train, df_val)
    print("Toy fine-tuning done.")

"""
def main():
    df_train = pd.read_pickle("data/processed/kp20k_train_preprocessed.pkl")
    df_val = pd.read_pickle("data/processed/kp20k_val_preprocessed.pkl")

    trainer = SciBARTFineTuner(
        SciBARTTrainerConfig(
            model_name="bloomberg/SciBART",   # 팀에서 사용하는 실제 HF model id로 교체 가능
            output_dir="outputs/scibart_ckpt",
            input_col="input_text",
            target_col="target_all_kps",      # "kp1 ; kp2 ; kp3" 형식
            max_source_length=512,
            max_target_length=128,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=5e-5,
            num_train_epochs=10,
            early_stopping_patience=2,
            early_stopping_threshold=0.0,
            fp16=False,
        )
    )

    trainer.train(df_train, df_val)
    print("Fine-tuning done.")
"""

if __name__ == "__main__":
    main()