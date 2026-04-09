"""
Embedder: SentenceTransformer 기반 임베딩 인터페이스.

지원 모델:
  - 'allenai/scibert_scivocab_uncased'        (학술 도메인 특화)
  - 'paraphrase-multilingual-MiniLM-L12-v2'   (다국어 경량 모델 / 기본 KeyBERT)
  - 또는 SentenceTransformer가 지원하는 임의 모델명
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    SentenceTransformer 래퍼.

    Args:
        model_name: HuggingFace 모델 ID 또는 SentenceTransformer 모델명
        device    : 'cuda' | 'cpu' | None (None이면 자동 감지)
    """

    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """
        텍스트 리스트를 임베딩 행렬로 변환.

        Args:
            texts    : 인코딩할 텍스트 리스트
            normalize: True면 L2 정규화 적용 (코사인 유사도 계산 최적화)

        Returns:
            shape (N, H) numpy 배열
        """
        return self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def encode_document(self, document: str) -> np.ndarray:
        """단일 문서를 (1, H) 배열로 인코딩."""
        return self.encode([document])

    def encode_candidates(self, candidates: list[str]) -> np.ndarray:
        """후보 키프레이즈 리스트를 (K, H) 배열로 인코딩."""
        return self.encode(candidates)
