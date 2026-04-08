from datasets import load_dataset

from src.training import SciBARTFineTuner, SciBARTTrainerConfig


def main():
    dataset = load_dataset(
        "arrow",
        data_files={
            "train": "./sampled_01_preprocessed/train/data-00000-of-00001.arrow",
            "validation": "./sampled_01_preprocessed/validation/data-00000-of-00001.arrow",
            "test": "./sampled_01_preprocessed/test/data-00000-of-00001.arrow",
        }
    )

    train_ds = dataset["train"]
    val_ds = dataset["validation"]

    print("Train sample:", train_ds[0])

    config = SciBARTTrainerConfig(
        model_name="uclanlp/scibart-large",
        output_dir="./outputs/scibart",
        batch_size=4,
        epochs=3,
    )

    trainer = SciBARTFineTuner(config)
    trainer.train(train_ds, val_ds)


if __name__ == "__main__":
    main()