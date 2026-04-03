from __future__ import annotations

import numpy as np
import pytest

from student_modeling.engine.embeddings import EmbeddingService


class TestEmbeddingService:
    @pytest.fixture(scope="class")
    def service(self):
        from student_modeling.config import get_settings

        settings = get_settings()
        api_key = settings.huggingface_api_key.get_secret_value()
        return EmbeddingService(api_key=api_key)

    def test_embed_returns_768_dim(self, service):
        vec = service.embed("Early Computing & Cryptography")
        assert vec.shape == (768,)
        assert vec.dtype == np.float32

    def test_embed_is_normalized(self, service):
        vec = service.embed("Programming Pioneers")
        assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-4)

    def test_embed_batch(self, service):
        texts = ["Early Computing", "Programming Pioneers", "Network Protocols"]
        vecs = service.embed_batch(texts)
        assert vecs.shape == (3, 768)

    def test_similar_texts_have_high_cosine(self, service):
        v1 = service.embed("machine learning algorithms")
        v2 = service.embed("deep learning neural networks")
        v3 = service.embed("cooking pasta recipes")
        sim_related = float(np.dot(v1, v2))
        sim_unrelated = float(np.dot(v1, v3))
        assert sim_related > sim_unrelated
