import os
import numpy as np
import datasets

def build_columns(df):
    df = df.copy()

    df["present_kps"] = df.apply(
        lambda r: [kp for kp, p in zip(r["keyphrases"], r["prmu"]) if p == "P"],
        axis=1
    )

    df["absent_kps"] = df.apply(
        lambda r: [kp for kp, p in zip(r["keyphrases"], r["prmu"]) if p != "P"],
        axis=1
    )

    df["source_text"] = df.apply(
        lambda r: f"generate keyphrases: title: {r['title']} abstract: {r['abstract']}",
        axis=1
    )

    df["target_text"] = df["keyphrases"].apply(lambda kps: str(np.array(kps)))

    return df

def save_to_arrow(df, path):
    os.makedirs(path, exist_ok=True)
    dataset = datasets.Dataset.from_pandas(df, preserve_index=False)
    dataset.save_to_disk(path)
    print(f"저장 완료: {path}")
