"""
Reranker: KeyBERT 코사인 유사도 1차 순위 + MMR 다양성 재순위.

KeyBERT 방식:
    score(doc, kp_i) = cosine_similarity(emb(doc), emb(kp_i))

MMR 공식:
    MMR_i = λ × sim(doc, c_i) − (1−λ) × max_{s∈S} sim(s, c_i)

λ=1.0 → 순수 relevance (MMR 비활성화)
λ=0.0 → 순수 diversity
λ=0.5~0.7 → 일반적으로 좋은 균형
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.embedder import Embedder


class Reranker:
    """
    KeyBERT + MMR 재순위 모듈.

    Args:
        embedder: Embedder 인스턴스
        top_k   : MMR이 최종 선택할 키프레이즈 수
        mmr_lambda: 기본 λ 값 (run 시 오버라이드 가능)
    """

    def __init__(self, embedder: Embedder, top_k: int = 10, mmr_lambda: float = 0.6):
        self.embedder = embedder
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda

    def keybert_rank(
        self,
        document: str,
        candidates: list[str],
    ) -> tuple[list[tuple[str, float]], np.ndarray, np.ndarray]:
        """
        KeyBERT 방식으로 후보 키프레이즈를 코사인 유사도 기준 1차 정렬.

        Returns:
            ranked_candidates : (키프레이즈, 유사도) 리스트 — 내림차순
            cand_embeddings   : 후보 임베딩 (K, H)
            doc_embedding     : 문서 임베딩 (1, H)
        """
        if not candidates:
            return [], np.array([]), np.array([])

        doc_embedding = self.embedder.encode_document(document)       # (1, H)
        cand_embeddings = self.embedder.encode_candidates(candidates)  # (K, H)

        similarities = cosine_similarity(doc_embedding, cand_embeddings)[0]  # (K,)
        ranked_indices = similarities.argsort()[::-1]
        ranked_candidates = [
            (candidates[i], float(similarities[i]))
            for i in ranked_indices
        ]

        return ranked_candidates, cand_embeddings, doc_embedding

    def mmr_rerank(
        self,
        doc_embedding: np.ndarray,
        candidates: list[str],
        cand_embeddings: np.ndarray,
        mmr_lambda: float = None,
        top_k: int = None,
    ) -> list[str]:
        """
        MMR(Maximal Marginal Relevance)으로 최종 키프레이즈 선택.

        알고리즘:
            1. 첫 선택: relevance 최대 후보
            2. 이후: MMR = λ×relevance − (1−λ)×max_redundancy 최대 후보 greedy 선택

        Args:
            doc_embedding   : 문서 임베딩 (1, H)
            candidates      : 후보 키프레이즈 리스트
            cand_embeddings : 후보 임베딩 (K, H)
            mmr_lambda      : λ 값. None이면 self.mmr_lambda 사용
            top_k           : 선택 수. None이면 self.top_k 사용

        Returns:
            선택된 키프레이즈 리스트 (최대 top_k개)
        """
        if mmr_lambda is None:
            mmr_lambda = self.mmr_lambda
        if top_k is None:
            top_k = self.top_k

        if not candidates:
            return []

        top_k = min(top_k, len(candidates))

        doc_sim = cosine_similarity(doc_embedding, cand_embeddings)[0]   # (K,)
        cand_sim = cosine_similarity(cand_embeddings, cand_embeddings)    # (K, K)

        selected_indices: list[int] = []
        remaining_indices = list(range(len(candidates)))

        for _ in range(top_k):
            if not remaining_indices:
                break

            if not selected_indices:
                best_idx = max(remaining_indices, key=lambda i: doc_sim[i])
            else:
                best_idx, best_score = None, -float('inf')
                for i in remaining_indices:
                    max_redundancy = max(cand_sim[i][j] for j in selected_indices)
                    mmr_score = mmr_lambda * doc_sim[i] - (1 - mmr_lambda) * max_redundancy
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = i

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [candidates[i] for i in selected_indices]

    def run(
        self,
        document: str,
        candidates: list[str],
        mmr_lambda: float = None,
        top_k: int = None,
    ) -> tuple[list[str], list[tuple[str, float]]]:
        """
        KeyBERT 1차 순위 → MMR 재순위를 한 번에 실행.

        Returns:
            final_kps     : MMR 최종 선택 키프레이즈 리스트
            ranked_before : KeyBERT 1차 순위 (키프레이즈, 유사도) 리스트
        """
        ranked_before, cand_embeddings, doc_embedding = self.keybert_rank(
            document, candidates
        )
        if not candidates:
            return [], []

        final_kps = self.mmr_rerank(
            doc_embedding=doc_embedding,
            candidates=candidates,
            cand_embeddings=cand_embeddings,
            mmr_lambda=mmr_lambda,
            top_k=top_k,
        )
        return final_kps, ranked_before
