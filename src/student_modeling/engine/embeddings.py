from __future__ import annotations

import numpy as np
from huggingface_hub import InferenceClient


class EmbeddingService:
    """Embeds text via the HuggingFace Inference API.

    Uses sentence-transformers/all-mpnet-base-v2 (768-dim) by default.
    All outputs are L2-normalized float32 arrays.
    """

    def __init__(self, api_key: str, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> None:
        self._client = InferenceClient(token=api_key)
        self._model = model_name

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns (768,) float32 array, L2-normalized."""
        vec = np.array(
            self._client.feature_extraction(text, model=self._model),
            dtype=np.float32,
        )
        if vec.ndim == 2:
            vec = vec.mean(axis=0)
        vec = vec / np.linalg.norm(vec)
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts. Returns (n, 768) float32 array, L2-normalized."""
        vecs = np.array(
            self._client.feature_extraction(texts, model=self._model),
            dtype=np.float32,
        )
        if vecs.ndim == 3:
            vecs = vecs.mean(axis=1)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        return vecs
