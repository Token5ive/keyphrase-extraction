import pandas as pd

from src.generation import CandidateGenerator, CandidateGeneratorConfig


def main():
    df_test = pd.read_pickle("data/processed/kp20k_test_toy_preprocessed.pkl")
    # df_test = pd.read_pickle("data/processed/kp20k_test_preprocessed.pkl")

    generator = CandidateGenerator(
        CandidateGeneratorConfig(
            model_path="outputs/scibart_ckpt_toy", # 실제로는 "outputs/scibart_ckpt" 이어야 함
            input_col="input_text",
            id_col="id",
            max_source_length=512,
            max_new_tokens=96,
            num_beams=50,             # 요청사항 반영
            num_return_sequences=20,  # 요청사항 반영
            batch_size=2,             # beam이 커서 batch는 작게 두는 걸 추천
            output_jsonl_path="data/candidates/kp20k_toy_test_candidates.jsonl",
            output_csv_path="data/candidates/kp20k_toy_test_candidates.csv",
            # output_jsonl_path="data/candidates/kp20k_test_candidates.jsonl",
            # output_csv_path="data/candidates/kp20k_test_candidates.csv",
        )
    )

    df_candidates = generator.generate_from_dataframe(df_test)
    generator.save_outputs(df_candidates)

    print(df_candidates.head())
    print("Candidate generation done.")


if __name__ == "__main__":
    main()