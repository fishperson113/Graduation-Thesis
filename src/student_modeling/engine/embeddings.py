from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Wraps sentence-transformers for embedding text into vectors.

    Uses all-mpnet-base-v2 (768-dim) by default, matching the GAM-RAG paper.
    All outputs are L2-normalized float32 arrays.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2") -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns (768,) float32 array, L2-normalized."""
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts. Returns (n, 768) float32 array, L2-normalized."""
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return vecs.astype(np.float32)
