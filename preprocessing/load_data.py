from datasets import load_dataset
import pandas as pd

def load_kp20k():
    dataset = load_dataset("taln-ls2n/kp20k")
    splits = {}
    for split in ['train', 'validation', 'test']:
        splits[split] = dataset[split].to_pandas()
        print(f"{split}: {len(splits[split])}개")
    return splits