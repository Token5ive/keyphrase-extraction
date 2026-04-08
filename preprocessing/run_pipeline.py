# preprocessing/pipeline.py

from .load_data import load_dataset_by_name
from .filter_data import run_preprocessing
from .build_column import build_columns, save_to_arrow


def run_pipeline(dataset="kp20k", save=True, base_save_path="output"):
    """
    전처리 전체 파이프라인 진입점.
    
    Args:
        dataset (str): 데이터셋 이름 ('kp20k' 또는 'inspec'). 기본값: 'kp20k'
        save (bool): Arrow 포맷으로 저장할지 여부. 기본값: True
        base_save_path (str): 저장 경로. 기본값: 'output'
    
    Returns:
        dict: split별 전처리된 DataFrame 딕셔너리
    
    Examples:
        # KP20k 전처리
        results = run_pipeline(dataset="kp20k")
        
        # Inspec 전처리 (평가용)
        results = run_pipeline(dataset="inspec", base_save_path="output/inspec")
    """
    print(f"=== {dataset.upper()} 데이터셋 전처리 시작 ===\n")
    
    # 데이터셋 로드
    splits = load_dataset_by_name(dataset)

    results = {}
    for split_name, df in splits.items():
        df_filtered = run_preprocessing(df, split=split_name)
        df_final = build_columns(df_filtered)

        if save:
            save_path = f"{base_save_path}/{dataset}_{split_name}_clean"
            save_to_arrow(df_final, save_path)

        results[split_name] = df_final

    return results
