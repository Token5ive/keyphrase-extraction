from src.evaluation.kp_metrics import KeyphraseEvaluator


def print_scores(title, result):
    print(f"\n[{title}]")
    print(f"Precision: {result.precision:.4f}")
    print(f"Recall:    {result.recall:.4f}")
    print(f"F1:        {result.f1:.4f}")


def main():
    evaluator = KeyphraseEvaluator()
    results = evaluator.load_json("./results/predictions.json")

    overall = evaluator.compute_macro_scores(
        results,
        pred_key="predicted_kps",
        gold_key="gold_target_kps",
    )

    present = evaluator.compute_macro_scores(
        results,
        pred_key="pred_present_kps",
        gold_key="gold_present_kps",
    )

    absent = evaluator.compute_macro_scores(
        results,
        pred_key="pred_absent_kps",
        gold_key="gold_absent_kps",
    )

    print_scores("Overall", overall)
    print_scores("Present", present)
    print_scores("Absent", absent)


if __name__ == "__main__":
    main()