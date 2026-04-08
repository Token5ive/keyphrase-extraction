from datasets import load_dataset
import pandas as pd


def load_kp20k():
    """KP20k 데이터셋 로드 (train, validation, test)"""
    dataset = load_dataset("taln-ls2n/kp20k")
    splits = {}
    for split in ['train', 'validation', 'test']:
        splits[split] = dataset[split].to_pandas()
        print(f"{split}: {len(splits[split])}개")
    return splits


def load_inspec():
    """Inspec 데이터셋 로드 (test만 존재)"""
    dataset = load_dataset("midas/inspec", "raw")
    splits = {}
    
    # Inspec은 test split만 있음
    if 'test' in dataset:
        splits['test'] = dataset['test'].to_pandas()
        print(f"test: {len(splits['test'])}개")
    
    return splits


def load_dataset_by_name(dataset_name="kp20k"):
    """
    데이터셋 이름으로 로드하는 통합 함수
    
    Args:
        dataset_name (str): 'kp20k' 또는 'inspec'
    
    Returns:
        dict: split별 DataFrame 딕셔너리
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "kp20k":
        return load_kp20k()
    elif dataset_name == "inspec":
        return load_inspec()
    else:
        raise ValueError(f"지원하지 않는 데이터셋: {dataset_name}. 'kp20k' 또는 'inspec'을 사용하세요.")