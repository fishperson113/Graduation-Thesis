from __future__ import annotations

import numpy as np
import pytest

from student_modeling.engine.kalman import (
    compute_gain,
    compute_residual,
    compute_time_decay,
    update_memory,
    update_perplexity,
)


class TestComputeGain:
    def test_high_perplexity_gives_high_gain(self):
        """When pi ~1 (new concept), K should be large (fast warm-up)."""
        K = compute_gain(pi=1.0, R=0.5)
        assert K == pytest.approx(1.0 / 1.5, abs=1e-6)
        assert K > 0.6

    def test_low_perplexity_gives_low_gain(self):
        """When pi is small (stable concept), K should be small (damped)."""
        K = compute_gain(pi=0.05, R=0.5)
        assert K == pytest.approx(0.05 / 0.55, abs=1e-6)
        assert K < 0.1

    def test_asymmetric_noise(self):
        """K_pos > K_neg for the same perplexity (correct answers learn faster)."""
        K_pos = compute_gain(pi=0.5, R=0.5)
        K_neg = compute_gain(pi=0.5, R=1.0)
        assert K_pos > K_neg

    def test_gain_is_between_0_and_1(self):
        for pi in [0.01, 0.1, 0.5, 0.9, 1.0]:
            for R in [0.5, 1.0]:
                K = compute_gain(pi, R)
                assert 0.0 < K < 1.0


class TestComputeResidual:
    def test_positive_residual_on_correct_with_low_alignment(self):
        """y=1 and memory far from query -> e > 0 (attraction)."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal(768).astype(np.float32)
        q /= np.linalg.norm(q)
        m = rng.standard_normal(768).astype(np.float32)
        m /= np.linalg.norm(m)
        e = compute_residual(y=1, query_emb=q, memory_vec=m)
        assert e > 0.5

    def test_negative_residual_on_incorrect_with_high_alignment(self):
        """y=0 and memory aligned with query -> e < 0 (repulsion)."""
        q = np.ones(768, dtype=np.float32)
        q /= np.linalg.norm(q)
        m = q.copy()
        e = compute_residual(y=0, query_emb=q, memory_vec=m)
        assert e == pytest.approx(-1.0, abs=1e-5)

    def test_residual_range(self):
        """Residual is in [-1, 2] since y in {0,1} and cos in [-1,1]."""
        q = np.ones(768, dtype=np.float32)
        q /= np.linalg.norm(q)
        m = -q.copy()
        e_pos = compute_residual(y=1, query_emb=q, memory_vec=m)
        assert e_pos == pytest.approx(2.0, abs=1e-5)


class TestUpdateMemory:
    def test_attraction_moves_toward_query(self):
        """Positive residual should increase cos(m_new, q)."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal(768).astype(np.float32)
        q /= np.linalg.norm(q)
        m = rng.standard_normal(768).astype(np.float32)
        m /= np.linalg.norm(m)
        cos_before = float(np.dot(m, q))
        m_new = update_memory(m_old=m, K=0.5, e=1.0, query_emb=q)
        cos_after = float(np.dot(m_new, q))
        assert cos_after > cos_before

    def test_repulsion_moves_away_from_query(self):
        """Negative residual should decrease cos(m_new, q)."""
        rng = np.random.default_rng(99)
        q = rng.standard_normal(768).astype(np.float32)
        q /= np.linalg.norm(q)
        # Start with m close to q but not identical (avoids collinear edge case)
        m = q + 0.1 * rng.standard_normal(768).astype(np.float32)
        m /= np.linalg.norm(m)
        cos_before = float(np.dot(m, q))
        m_new = update_memory(m_old=m, K=0.5, e=-0.5, query_emb=q)
        cos_after = float(np.dot(m_new, q))
        assert cos_after < cos_before

    def test_output_is_normalized(self):
        """Memory vector must have unit norm after update."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal(768).astype(np.float32)
        q /= np.linalg.norm(q)
        m = rng.standard_normal(768).astype(np.float32)
        m /= np.linalg.norm(m)
        m_new = update_memory(m_old=m, K=0.8, e=1.5, query_emb=q)
        assert np.linalg.norm(m_new) == pytest.approx(1.0, abs=1e-5)


class TestUpdatePerplexity:
    def test_perplexity_decreases_after_update(self):
        """After a feedback event, perplexity should decrease (confidence grows)."""
        pi_new = update_perplexity(pi_old=1.0, K=0.667, Q=0.05)
        assert pi_new < 1.0

    def test_perplexity_floor(self):
        """Perplexity never drops below Q."""
        pi = 1.0
        for _ in range(100):
            pi = update_perplexity(pi_old=pi, K=0.9, Q=0.05)
        assert pi >= 0.05

    def test_perplexity_clipped_to_0_1(self):
        pi = update_perplexity(pi_old=0.99, K=0.01, Q=0.05)
        assert 0.0 <= pi <= 1.0

    def test_low_gain_preserves_perplexity(self):
        """Low K means perplexity changes slowly."""
        pi_new = update_perplexity(pi_old=0.5, K=0.05, Q=0.05)
        assert pi_new == pytest.approx((1 - 0.05) * 0.5 + 0.05, abs=1e-6)


class TestComputeTimeDecay:
    def test_no_time_elapsed(self):
        """Zero days -> no decay."""
        pi = compute_time_decay(pi=0.1, days_elapsed=0.0, decay_rate=0.02)
        assert pi == pytest.approx(0.1, abs=1e-6)

    def test_decay_increases_perplexity(self):
        """Time passing increases perplexity (forgetting)."""
        pi = compute_time_decay(pi=0.1, days_elapsed=10.0, decay_rate=0.02)
        assert pi > 0.1
        assert pi == pytest.approx(0.1 + 0.02 * 10.0, abs=1e-6)

    def test_decay_clipped_at_1(self):
        """Perplexity cannot exceed 1.0."""
        pi = compute_time_decay(pi=0.9, days_elapsed=100.0, decay_rate=0.02)
        assert pi == 1.0
