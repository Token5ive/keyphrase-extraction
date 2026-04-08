from datasets import load_dataset

from src.generation import CandidateGenerator, CandidateGeneratorConfig


def main():
    dataset = load_dataset(
        "arrow",
        data_files={
            "test": "./sampled_01_preprocessed/test/data-00000-of-00001.arrow"
        }
    )

    test_ds = dataset["test"]

    generator = CandidateGenerator(
        CandidateGeneratorConfig(
            model_path="./outputs/scibart",
            input_col="source_text",
            id_col="id",
            batch_size=4,
        )
    )

    results = generator.generate(test_ds)
    generator.save(results)

    print("Generation 완료")


if __name__ == "__main__":
    main()