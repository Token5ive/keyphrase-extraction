"""
main.py — KPE 실험 실행 스크립트

실험 구성 (CLAUDE.md 기준):
  Exp 1. Preprocessing 적용 vs 미적용 성능 비교
  Exp 2. Embedding 모델 비교 (SciBERT vs MiniLM)
  Exp 3. Lambda Sweep [0.0, 0.1, ..., 1.0]
  Exp 4. Qualitative Analysis — 특정 샘플 정성 분석

결과는 results/<실험명>/ 에 저장됩니다.
"""

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch

import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.embedder import Embedder
from src.reranker import Reranker
from src.evaluator import Evaluator, print_results_table, show_example


# ── 기본 설정 ──────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    'data_path':    'data/predictions.json',
    'num_samples':  None,           # None = 전체 189개 사용

    # 전처리
    'min_words': 1,
    'max_words': 5,

    # 임베딩
    'encoder_scibert': 'allenai/scibert_scivocab_uncased',
    'encoder_mpnet':   'sentence-transformers/all-mpnet-base-v2',
    'encoder_minilm':  'all-MiniLM-L6-v2',                       # KeyBERT 기본 모델

    # MMR
    'mmr_lambda':  0.6,
    'top_k':       10,

    # 평가
    'eval_k': 5,

    # Lambda sweep 범위
    'lambda_values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

    # 정성 분석 샘플 인덱스
    'qualitative_indices': [0, 1, 2],
}


# ── 유틸 ───────────────────────────────────────────────────────────────

def _save_results(df: pd.DataFrame, out_dir: str, name: str) -> None:
    """DataFrame을 CSV와 마크다운 테이블로 저장."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f'{name}.csv')
    md_path  = os.path.join(out_dir, f'{name}.md')
    df.to_csv(csv_path)
    df.to_markdown(md_path)
    print(f'  → 저장: {csv_path}')


def _build_reranker(encoder_model: str, cfg: dict, device: str) -> Reranker:
    print(f'  Loading encoder: {encoder_model} ...')
    embedder = Embedder(encoder_model, device=device)
    return Reranker(embedder, top_k=cfg['top_k'], mmr_lambda=cfg['mmr_lambda'])


def _load_records(cfg: dict) -> list[dict]:
    loader = DataLoader(cfg['data_path'])
    records = loader.load(num_samples=cfg['num_samples'])
    print(f'  Loaded {len(records)} samples from {cfg["data_path"]}')
    return records


# ── 실험 함수 ──────────────────────────────────────────────────────────

def exp1_preprocessing(cfg: dict, device: str) -> None:
    """
    Exp 1: Preprocessing 적용 vs 미적용 비교.
    동일 encoder(SciBERT), 동일 λ 사용.
    """
    print('\n' + '='*60)
    print('Exp 1: Preprocessing 적용 vs 미적용')
    print('='*60)
    out_dir = 'results/exp1_preprocessing'

    preprocessor = Preprocessor(cfg['min_words'], cfg['max_words'])
    reranker = _build_reranker(cfg['encoder_scibert'], cfg, device)
    evaluator = Evaluator(reranker, eval_k=cfg['eval_k'])
    records_raw = _load_records(cfg)

    results = {}

    for use_prep, label in [(False, 'Raw (no preprocessing)'), (True, 'Preprocessed')]:
        records = preprocessor.apply_to_records(records_raw, use_preprocess=use_prep)
        before = sum(len(r['candidates']) for r in records_raw) / len(records_raw)
        after  = sum(len(r['candidates']) for r in records) / len(records)
        print(f'\n[{label}] 평균 후보 수: {after:.1f} (raw: {before:.1f})')

        # Baseline
        baseline = evaluator.run_baseline(records, desc=f'{label} - Baseline')
        results[f'{label} | Baseline'] = baseline

        # KeyBERT+MMR
        mmr = evaluator.run_pipeline(
            records, mmr_lambda=cfg['mmr_lambda'],
            desc=f'{label} - KeyBERT+MMR'
        )
        results[f'{label} | KeyBERT+MMR (λ={cfg["mmr_lambda"]})'] = mmr

    print_results_table(results, cfg['eval_k'])
    df = pd.DataFrame(results).T
    _save_results(df, out_dir, 'exp1_preprocessing')


def exp2_encoder(cfg: dict, device: str) -> None:
    """
    Exp 2: SciBERT vs all-mpnet-base-v2 vs MiniLM-L6-v2(KeyBERT 기본) 비교.
    Preprocessing 적용, λ=CONFIG['mmr_lambda'] 고정.
    """
    print('\n' + '='*60)
    print('Exp 2: Embedding 모델 비교 (SciBERT vs MPNet vs MiniLM)')
    print('='*60)
    out_dir = 'results/exp2_encoder'

    preprocessor = Preprocessor(cfg['min_words'], cfg['max_words'])
    records_raw = _load_records(cfg)
    records = preprocessor.apply_to_records(records_raw, use_preprocess=True)

    results = {}
    models = {
        'SciBERT': cfg['encoder_scibert'],
        'MPNet':   cfg['encoder_mpnet'],
        'MiniLM (KeyBERT)': cfg['encoder_minilm'],
    }

    for model_label, model_name in models.items():
        reranker = _build_reranker(model_name, cfg, device)
        evaluator = Evaluator(reranker, eval_k=cfg['eval_k'])

        baseline = evaluator.run_baseline(records, desc=f'{model_label} - Baseline')
        results[f'{model_label} | Baseline'] = baseline

        keybert = evaluator.run_pipeline(
            records, mmr_lambda=1.0, desc=f'{model_label} - KeyBERT-only'
        )
        results[f'{model_label} | KeyBERT-only (λ=1.0)'] = keybert

        mmr = evaluator.run_pipeline(
            records, mmr_lambda=cfg['mmr_lambda'],
            desc=f'{model_label} - KeyBERT+MMR'
        )
        results[f'{model_label} | KeyBERT+MMR (λ={cfg["mmr_lambda"]})'] = mmr

    print_results_table(results, cfg['eval_k'])
    df = pd.DataFrame(results).T
    _save_results(df, out_dir, 'exp2_encoder')


def exp3_lambda_sweep(cfg: dict, device: str) -> None:
    """
    Exp 3: λ 값을 [0.0, 0.1, ..., 1.0]으로 변경하며 성능 추이 기록.
    SciBERT encoder, Preprocessing 적용.
    """
    print('\n' + '='*60)
    print('Exp 3: Lambda Sweep')
    print('='*60)
    out_dir = 'results/exp3_lambda_sweep'

    preprocessor = Preprocessor(cfg['min_words'], cfg['max_words'])
    reranker = _build_reranker(cfg['encoder_scibert'], cfg, device)
    evaluator = Evaluator(reranker, eval_k=cfg['eval_k'])

    records_raw = _load_records(cfg)
    records = preprocessor.apply_to_records(records_raw, use_preprocess=True)

    sweep = evaluator.lambda_sweep(records, cfg['lambda_values'])

    # 결과 테이블
    rows = []
    for lam, res in sweep.items():
        rows.append({'lambda': lam, **res})
    df = pd.DataFrame(rows).set_index('lambda')
    print('\n', df.to_string())
    _save_results(df, out_dir, 'exp3_lambda_sweep')

    # 최적 λ
    K = cfg['eval_k']
    best_lam = max(sweep, key=lambda l: sweep[l][f'F1@{K}'])
    print(f'\n  Best λ = {best_lam} (F1@{K} = {sweep[best_lam][f"F1@{K}"]:.4f})')

    # 시각화
    lams = list(sweep.keys())
    f1k_vals = [sweep[l][f'F1@{K}'] for l in lams]
    f1m_vals = [sweep[l]['F1@M']     for l in lams]
    ild_vals  = [sweep[l]['ILD']      for l in lams]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(lams, f1k_vals, marker='o', label=f'F1@{K}')
    axes[0].plot(lams, f1m_vals, marker='s', label='F1@M')
    axes[0].set_xlabel('λ')
    axes[0].set_ylabel('F1')
    axes[0].set_title(f'F1 vs λ (SciBERT, preprocessed)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(best_lam, color='red', linestyle='--', alpha=0.5, label=f'best λ={best_lam}')
    axes[0].legend()

    axes[1].plot(lams, ild_vals, marker='^', color='green', label='ILD')
    axes[1].set_xlabel('λ')
    axes[1].set_ylabel('ILD')
    axes[1].set_title('Diversity (ILD) vs λ')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, 'exp3_lambda_sweep.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f'  → 그래프 저장: {fig_path}')


def exp4_qualitative(cfg: dict, device: str) -> None:
    """
    Exp 4: 특정 샘플 정성 분석 — Ground Truth vs KeyBERT vs MMR 결과 비교.
    """
    print('\n' + '='*60)
    print('Exp 4: Qualitative Analysis')
    print('='*60)
    out_dir = 'results/exp4_qualitative'
    os.makedirs(out_dir, exist_ok=True)

    preprocessor = Preprocessor(cfg['min_words'], cfg['max_words'])
    reranker = _build_reranker(cfg['encoder_scibert'], cfg, device)

    records_raw = _load_records(cfg)
    records = preprocessor.apply_to_records(records_raw, use_preprocess=True)

    log_lines = []
    for idx in cfg['qualitative_indices']:
        if idx >= len(records):
            print(f'  인덱스 {idx}가 범위를 벗어남 (총 {len(records)}개). 건너뜀.')
            continue

        header = f'\n{"="*65}\n[Example {idx}]\n{"="*65}'
        print(header)
        log_lines.append(header)

        import io, sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        show_example(records[idx], reranker, cfg['mmr_lambda'], cfg['eval_k'])
        sys.stdout = old_stdout
        output = buf.getvalue()
        print(output)
        log_lines.append(output)

    log_path = os.path.join(out_dir, 'qualitative_analysis.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    print(f'\n  → 정성 분석 결과 저장: {log_path}')


# ── 메인 ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='KPE 실험 스크립트')
    parser.add_argument(
        '--exp', nargs='+',
        choices=['1', '2', '3', '4', 'all'],
        default=['all'],
        help='실행할 실험 번호 (1 2 3 4 또는 all)',
    )
    parser.add_argument('--data_path',   default=DEFAULT_CONFIG['data_path'])
    parser.add_argument('--num_samples', type=int,   default=DEFAULT_CONFIG['num_samples'])
    parser.add_argument('--eval_k',      type=int,   default=DEFAULT_CONFIG['eval_k'])
    parser.add_argument('--top_k',       type=int,   default=DEFAULT_CONFIG['top_k'])
    parser.add_argument('--mmr_lambda',  type=float, default=DEFAULT_CONFIG['mmr_lambda'])
    parser.add_argument('--min_words',   type=int,   default=DEFAULT_CONFIG['min_words'])
    parser.add_argument('--max_words',   type=int,   default=DEFAULT_CONFIG['max_words'])
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = dict(DEFAULT_CONFIG)
    cfg.update({
        'data_path':   args.data_path,
        'num_samples': args.num_samples,
        'eval_k':      args.eval_k,
        'top_k':       args.top_k,
        'mmr_lambda':  args.mmr_lambda,
        'min_words':   args.min_words,
        'max_words':   args.max_words,
    })

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f'Device: {device}')
    print(f'Config: {json.dumps({k: v for k, v in cfg.items() if k != "lambda_values"}, indent=2, ensure_ascii=False)}')

    exps = set(args.exp)
    run_all = 'all' in exps

    t0 = time.time()

    if run_all or '1' in exps:
        exp1_preprocessing(cfg, device)

    if run_all or '2' in exps:
        exp2_encoder(cfg, device)

    if run_all or '3' in exps:
        exp3_lambda_sweep(cfg, device)

    if run_all or '4' in exps:
        exp4_qualitative(cfg, device)

    elapsed = time.time() - t0
    print(f'\n전체 실험 완료: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
