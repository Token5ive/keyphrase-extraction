# preprocessing/pipeline.py

from .load_data import load_kp20k
from .filter_data import run_preprocessing
from .build_column import build_columns, save_to_arrow

def run_pipeline(save=True, base_save_path="output"):
    """
    전처리 전체 파이프라인 진입점.
    다른 팀원이 여기서만 호출하면 됨.
    """
    splits = load_kp20k()

    results = {}
    for split_name, df in splits.items():
        df_filtered = run_preprocessing(df, split=split_name)
        df_final = build_columns(df_filtered)

        if save:
            save_to_arrow(df_final, f"{base_save_path}/{split_name}_clean")

        results[split_name] = df_final  # 저장 안 하고 df로 바로 넘길 수도 있음

    return results
