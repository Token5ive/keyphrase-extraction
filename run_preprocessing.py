from datasets import load_dataset
from src.preprocess import KP20KPreprocessor, KP20KPreprocessorConfig
import os

os.makedirs("data/processed", exist_ok=True)

def main():
    dataset = load_dataset("taln-ls2n/kp20k")

    use_toy = True          # 전체 데이터셋 ; false , toy dataset ; true
    toy_size = 5000

    if use_toy:
        train_split = dataset["train"].select(range(min(toy_size, len(dataset["train"]))))
        val_split = dataset["validation"].select(range(min(toy_size, len(dataset["validation"]))))
        test_split = dataset["test"].select(range(min(toy_size, len(dataset["test"]))))
    else:
        train_split = dataset["train"]
        val_split = dataset["validation"]
        test_split = dataset["test"]

    df_train = train_split.to_pandas()
    df_val = val_split.to_pandas()
    df_test = test_split.to_pandas()

    preprocessor = KP20KPreprocessor(
        KP20KPreprocessorConfig(
            sep_token="[SEP]",
            kp_sep_token=" ; ",
            lowercase=True,
            use_prmu=False,
            save_stemmed_columns=False,
            max_abstract_words=512,
            max_keyphrases=15,
            drop_duplicates=True,
        )
    )

    df_train_proc, df_val_proc, df_test_proc = preprocessor.preprocess_splits(
        df_train, df_val, df_test
    )

    suffix = "toy" if use_toy else "full"

    df_train_proc.to_pickle(f"data/processed/kp20k_train_{suffix}_preprocessed.pkl")
    df_val_proc.to_pickle(f"data/processed/kp20k_val_{suffix}_preprocessed.pkl")
    df_test_proc.to_pickle(f"data/processed/kp20k_test_{suffix}_preprocessed.pkl")

    print(f"{suffix} preprocessing done.")
    print("train:", df_train_proc.shape)
    print("val:", df_val_proc.shape)
    print("test:", df_test_proc.shape)


if __name__ == "__main__":
    main()