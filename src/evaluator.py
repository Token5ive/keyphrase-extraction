"""
Evaluator: Stemming 기반 평가 지표 계산 및 파이프라인 실행.

Core Metrics (Stemming 적용):
  P@K, R@K, F1@K  — 상위 K개 예측 기준
  P@M, R@M, F1@M  — M = gold keyphrase 수 (샘플마다 다름)

Diversity / Quality:
  ILD  (Intra-List Diversity) — 예측 KP 간 평균 쌍별 코사인 거리
  SR   (Success Rate)         — 상위 K개 중 정답 포함 여부

Present / Absent 분석:
  F1@K(Present), F1@K(Absent) — gold KP의 문서 등장 여부 기준 분리 평가
"""

import time
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from src.reranker import Reranker

stemmer = PorterStemmer()


# ── 기본 유틸 ──────────────────────────────────────────────────────────

def stem(phrase: str) -> str:
    """Porter Stemming으로 어형 정규화 (소문자 + 어간 추출)."""
    return ' '.join(stemmer.stem(w) for w in phrase.lower().split())


def prf_at_k(preds: list[str], golds: list[str], k: int) -> tuple[float, float, float]:
    """
    P@K, R@K, F1@K 반환 (Stemming 적용).

    중복 주의: preds[:k] 내에 stem이 동일한 표현이 여러 개 있을 수 있음
    (예: 'nonlocal game' / 'nonlocal games' → 모두 'nonloc game').
    set 교집합으로 중복을 제거하여 match를 계산해야
    recall > 1.0, F1 > 1.0 같은 수학적으로 불가능한 값을 방지할 수 있음.

    Returns:
        (precision, recall, f1) — 각 0.0 ~ 1.0
    """
    pred_set = {stem(p) for p in preds[:k]}   # set: stem 중복 제거
    gold_set = {stem(g) for g in golds}

    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0

    match = len(pred_set & gold_set)          # set 교집합 — 중복 없이 카운팅
    precision = match / k
    recall = match / len(gold_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def ild_score(keyphrases: list[str], embedder) -> float:
    """
    Intra-List Diversity: 예측 KP 간 평균 쌍별 코사인 거리.

    ILD = mean_{i≠j} (1 − cos_sim(emb_i, emb_j))
    값이 높을수록 다양한 키프레이즈 집합.
    """
    if len(keyphrases) < 2:
        return 0.0
    embs = embedder.encode(keyphrases, normalize=True)
    sim_matrix = cosine_similarity(embs)
    n = len(keyphrases)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean([1.0 - sim_matrix[i][j] for i, j in pairs]))


def split_present_absent(golds: list[str], document: str) -> tuple[list[str], list[str]]:
    """
    gold KP를 Present / Absent로 분리.

    Present : 키프레이즈(소문자)가 문서(소문자)에 부분 문자열로 등장
    Absent  : 나머지
    """
    doc_lower = document.lower()
    present, absent = [], []
    for g in golds:
        (present if g.lower() in doc_lower else absent).append(g)
    return present, absent


def _nanmean(lst: list) -> float:
    arr = [x for x in lst if not (isinstance(x, float) and np.isnan(x))]
    return round(float(np.mean(arr)), 4) if arr else float('nan')


def _summarize(acc: dict, k: int) -> dict:
    """누산기 dict → 지표 요약 dict."""
    return {
        f'P@{k}':            round(float(np.mean(acc['p_k'])),  4),
        f'R@{k}':            round(float(np.mean(acc['r_k'])),  4),
        f'F1@{k}':           round(float(np.mean(acc['f1_k'])), 4),
        'P@M':               round(float(np.mean(acc['p_m'])),   4),
        'R@M':               round(float(np.mean(acc['r_m'])),   4),
        'F1@M':              round(float(np.mean(acc['f1_m'])),  4),
        'ILD':               round(float(np.mean(acc['ild'])),   4),
        'SR':                round(float(np.mean(acc['sr'])),    4),
        'Runtime(ms)':       round(float(np.mean(acc['runtime_ms'])), 2),
        f'F1@{k}(Present)':  _nanmean(acc['f1_present']),
        f'F1@{k}(Absent)':   _nanmean(acc['f1_absent']),
    }


def _empty_acc() -> dict:
    return {k: [] for k in [
        'p_k', 'r_k', 'f1_k', 'p_m', 'r_m', 'f1_m',
        'ild', 'sr', 'runtime_ms', 'f1_present', 'f1_absent',
    ]}


# ── 파이프라인 실행 함수 ────────────────────────────────────────────────

class Evaluator:
    """
    KeyBERT+MMR 파이프라인을 레코드 리스트 전체에 대해 실행하고 평가.

    Args:
        reranker : Reranker 인스턴스
        eval_k   : F1@K 에서 K
    """

    def __init__(self, reranker: Reranker, eval_k: int = 5):
        self.reranker = reranker
        self.eval_k = eval_k

    def run_pipeline(
        self,
        records: list[dict],
        mmr_lambda: float,
        top_k: int = None,
        desc: str = '',
    ) -> dict:
        """
        KeyBERT 순위 → MMR → 평가를 레코드 전체에 실행.

        Args:
            records    : DataLoader가 반환한 레코드 리스트
            mmr_lambda : MMR λ 값
            top_k      : MMR 선택 수. None이면 Reranker 기본값 사용
            desc       : tqdm 진행 표시 레이블

        Returns:
            지표 요약 dict (P@K, R@K, F1@K, P@M, R@M, F1@M, ILD, SR, ...)
        """
        eval_k = self.eval_k
        acc = _empty_acc()

        label = desc or f'λ={mmr_lambda:.2f}'
        for rec in tqdm(records, desc=label, leave=False):
            document = rec['document']
            candidates = rec['candidates']
            golds = rec['keyphrases']

            if not candidates:
                for key in acc:
                    acc[key].append(0.0)
                continue

            t_start = time.time()

            final_kps, _ = self.reranker.run(
                document=document,
                candidates=candidates,
                mmr_lambda=mmr_lambda,
                top_k=top_k,
            )

            elapsed_ms = (time.time() - t_start) * 1000

            p_k, r_k, f1_k = prf_at_k(final_kps, golds, k=eval_k)
            p_m, r_m, f1_m = prf_at_k(final_kps, golds, k=len(golds))
            ild = ild_score(final_kps, self.reranker.embedder)

            gold_stems = {stem(g) for g in golds}
            sr = 1.0 if any(stem(p) in gold_stems for p in final_kps[:eval_k]) else 0.0

            present_golds, absent_golds = split_present_absent(golds, document)
            f1_pre = prf_at_k(final_kps, present_golds, k=eval_k)[2] if present_golds else float('nan')
            f1_abs = prf_at_k(final_kps, absent_golds,  k=eval_k)[2] if absent_golds  else float('nan')

            acc['p_k'].append(p_k);    acc['r_k'].append(r_k);    acc['f1_k'].append(f1_k)
            acc['p_m'].append(p_m);    acc['r_m'].append(r_m);    acc['f1_m'].append(f1_m)
            acc['ild'].append(ild);    acc['sr'].append(sr)
            acc['runtime_ms'].append(elapsed_ms)
            acc['f1_present'].append(f1_pre)
            acc['f1_absent'].append(f1_abs)

        return _summarize(acc, eval_k)

    def run_baseline(
        self,
        records: list[dict],
        desc: str = 'Baseline',
    ) -> dict:
        """
        KeyBART 생성 순서 그대로 (reranking 없음) 평가.
        ILD 계산에는 embedder를 사용.
        """
        eval_k = self.eval_k
        acc = _empty_acc()

        for rec in tqdm(records, desc=desc, leave=False):
            document = rec['document']
            golds = rec['keyphrases']
            preds = rec['candidates']
            t0 = time.time()

            p_k, r_k, f1_k = prf_at_k(preds, golds, k=eval_k)
            p_m, r_m, f1_m = prf_at_k(preds, golds, k=len(golds))
            ild = ild_score(preds[:eval_k], self.reranker.embedder)

            gold_stems = {stem(g) for g in golds}
            sr = 1.0 if any(stem(p) in gold_stems for p in preds[:eval_k]) else 0.0

            present_g, absent_g = split_present_absent(golds, document)
            f1_pre = prf_at_k(preds, present_g, k=eval_k)[2] if present_g else float('nan')
            f1_abs = prf_at_k(preds, absent_g,  k=eval_k)[2] if absent_g  else float('nan')
            elapsed_ms = (time.time() - t0) * 1000

            acc['p_k'].append(p_k);    acc['r_k'].append(r_k);    acc['f1_k'].append(f1_k)
            acc['p_m'].append(p_m);    acc['r_m'].append(r_m);    acc['f1_m'].append(f1_m)
            acc['ild'].append(ild);    acc['sr'].append(sr)
            acc['runtime_ms'].append(elapsed_ms)
            acc['f1_present'].append(f1_pre)
            acc['f1_absent'].append(f1_abs)

        return _summarize(acc, eval_k)

    def lambda_sweep(
        self,
        records: list[dict],
        lambda_values: list[float],
        top_k: int = None,
    ) -> dict[float, dict]:
        """
        여러 λ 값에 대해 run_pipeline을 순차 실행.

        Returns:
            {lambda: 지표_dict} 형태의 dict
        """
        results = {}
        for lam in lambda_values:
            results[lam] = self.run_pipeline(
                records, mmr_lambda=lam, top_k=top_k, desc=f'λ={lam:.1f}'
            )
        return results

    def run_score_fusion(
        self,
        records: list[dict],
        alpha: float,
        mmr_lambda: float,
        top_k: int = None,
        desc: str = '',
    ) -> dict:
        """
        Score Fusion + MMR 파이프라인을 레코드 전체에 실행.

        각 후보의 재순위 점수:
            fusion_score(i) = α × beam_score(i) + (1−α) × cosine_sim(doc, c_i)
        MMR relevance 항에 fusion_score를 사용.

        Args:
            records    : DataLoader + Preprocessor를 거친 레코드 리스트
                         (candidate_positions, n_candidates_original 필드 필요)
            alpha      : beam score 가중치 (0.0=cosine only, 1.0=beam only)
            mmr_lambda : MMR λ 값
            top_k      : MMR 선택 수. None이면 Reranker 기본값 사용
            desc       : tqdm 진행 표시 레이블

        Returns:
            지표 요약 dict
        """
        eval_k = self.eval_k
        acc = _empty_acc()

        label = desc or f'α={alpha:.2f} λ={mmr_lambda:.2f}'
        for rec in tqdm(records, desc=label, leave=False):
            document  = rec['document']
            candidates = rec['candidates']
            positions  = rec.get('candidate_positions', list(range(len(candidates))))
            n_original = rec.get('n_candidates_original', len(candidates))
            golds      = rec['keyphrases']

            if not candidates:
                for key in acc:
                    acc[key].append(0.0)
                continue

            t_start = time.time()

            final_kps, _ = self.reranker.run_score_fusion(
                document=document,
                candidates=candidates,
                positions=positions,
                n_original=n_original,
                alpha=alpha,
                mmr_lambda=mmr_lambda,
                top_k=top_k,
            )

            elapsed_ms = (time.time() - t_start) * 1000

            p_k, r_k, f1_k = prf_at_k(final_kps, golds, k=eval_k)
            p_m, r_m, f1_m = prf_at_k(final_kps, golds, k=len(golds))
            ild = ild_score(final_kps, self.reranker.embedder)

            gold_stems = {stem(g) for g in golds}
            sr = 1.0 if any(stem(p) in gold_stems for p in final_kps[:eval_k]) else 0.0

            present_golds, absent_golds = split_present_absent(golds, document)
            f1_pre = prf_at_k(final_kps, present_golds, k=eval_k)[2] if present_golds else float('nan')
            f1_abs = prf_at_k(final_kps, absent_golds,  k=eval_k)[2] if absent_golds  else float('nan')

            acc['p_k'].append(p_k);    acc['r_k'].append(r_k);    acc['f1_k'].append(f1_k)
            acc['p_m'].append(p_m);    acc['r_m'].append(r_m);    acc['f1_m'].append(f1_m)
            acc['ild'].append(ild);    acc['sr'].append(sr)
            acc['runtime_ms'].append(elapsed_ms)
            acc['f1_present'].append(f1_pre)
            acc['f1_absent'].append(f1_abs)

        return _summarize(acc, eval_k)

    def alpha_sweep(
        self,
        records: list[dict],
        alpha_values: list[float],
        mmr_lambda: float,
        top_k: int = None,
    ) -> dict[float, dict]:
        """
        여러 α 값에 대해 run_score_fusion을 순차 실행.

        Returns:
            {alpha: 지표_dict} 형태의 dict
        """
        results = {}
        for alpha in alpha_values:
            results[alpha] = self.run_score_fusion(
                records,
                alpha=alpha,
                mmr_lambda=mmr_lambda,
                top_k=top_k,
                desc=f'α={alpha:.1f}',
            )
        return results


# ── 출력 유틸 ──────────────────────────────────────────────────────────

def print_results_table(methods: dict[str, dict], eval_k: int) -> None:
    """
    실험 결과를 테이블 형식으로 출력.

    Args:
        methods : {'레이블': result_dict, ...} 순서 보장 dict
        eval_k  : 헤더 생성용 (F1@K 등)
    """
    K = eval_k
    headers = [
        f'P@{K}', f'R@{K}', f'F1@{K}',
        'P@M', 'R@M', 'F1@M',
        'ILD', 'SR', 'Runtime(ms)',
        f'F1@{K}(Present)', f'F1@{K}(Absent)',
    ]
    col_w = 14
    print(f'\n{"Method":<30}', end='')
    for h in headers:
        print(f'  {h:>{col_w}}', end='')
    print()
    print('-' * (30 + (col_w + 2) * len(headers)))
    for method, res in methods.items():
        print(f'{method:<30}', end='')
        for h in headers:
            val = res.get(h, float('nan'))
            print(
                f'  {val:>{col_w}.4f}' if isinstance(val, float)
                else f'  {str(val):>{col_w}}',
                end=''
            )
        print()


def show_example(
    rec: dict,
    reranker: Reranker,
    mmr_lambda: float,
    eval_k: int = 5,
) -> None:
    """
    단일 샘플의 키프레이즈 추출 과정을 단계별로 출력.

    표시:
        - Gold keyphrases (정답)
        - KeyBERT 1차 순위 (유사도 점수 포함)
        - MMR 최종 선택 (다양성 반영)
    """
    document = rec['document']
    candidates = rec['candidates']
    golds = rec['keyphrases']

    ranked_before, cand_embeddings, doc_embedding = reranker.keybert_rank(document, candidates)
    final_kps = reranker.mmr_rerank(
        doc_embedding=doc_embedding,
        candidates=candidates,
        cand_embeddings=cand_embeddings,
        mmr_lambda=mmr_lambda,
        top_k=eval_k,
    )

    gold_stems = {stem(g) for g in golds}

    print(f'Title   : {rec["title"]}')
    print(f'Gold KPs: {golds}\n')

    print(f'[KeyBERT Top-{eval_k}] (코사인 유사도 기준, before MMR)')
    for kp, score in ranked_before[:eval_k]:
        hit = '✓' if stem(kp) in gold_stems else ' '
        print(f'  {hit} [{score:.4f}] {kp}')

    print(f'\n[MMR Top-{eval_k}] (λ={mmr_lambda}, diversity 반영)')
    for kp in final_kps:
        hit = '✓' if stem(kp) in gold_stems else ' '
        print(f'  {hit} {kp}')

    f1k = prf_at_k(final_kps, golds, k=eval_k)[2]
    f1m = prf_at_k(final_kps, golds, k=len(golds))[2]
    print(f'\n  F1@{eval_k}={f1k:.4f}  F1@M={f1m:.4f}')
