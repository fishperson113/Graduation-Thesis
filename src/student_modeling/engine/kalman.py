from __future__ import annotations

from datetime import datetime

import numpy as np


def compute_gain(pi: float, R: float) -> float:
    """Kalman gain: K = pi / (pi + R).

    High pi (uncertain) -> high K (fast update / warm-up).
    Low pi (stable) -> low K (damped refinement).
    """
    return pi / (pi + R)


def compute_residual(y: int, query_emb: np.ndarray, memory_vec: np.ndarray) -> float:
    """Residual: e = y - cos(q, m_old).

    e > 0 -> pull memory toward query (attraction).
    e < 0 -> push memory away from query (repulsion).
    """
    cos_sim = float(np.dot(query_emb, memory_vec))
    return y - cos_sim


def update_memory(
    m_old: np.ndarray, K: float, e: float, query_emb: np.ndarray
) -> np.ndarray:
    """Update memory vector: m_new = normalize(m_old + K * e * q).

    Preserves components orthogonal to q, adjusts projected support.
    Post-update L2 normalization ensures ||m|| = 1.
    """
    m_new = m_old + K * e * query_emb
    norm = np.linalg.norm(m_new)
    if norm < 1e-10:
        return m_old.copy()
    return m_new / norm


def update_perplexity(pi_old: float, K: float, Q: float) -> float:
    """Update perplexity: pi_new = clip[0,1]((1 - K) * pi_old + Q).

    Process noise Q prevents pi from collapsing to 0.
    """
    pi_new = (1.0 - K) * pi_old + Q
    return max(0.0, min(1.0, pi_new))


def compute_time_decay(pi: float, days_elapsed: float, decay_rate: float) -> float:
    """Increase perplexity based on time since last interaction.

    Models the forgetting curve: longer absence -> higher uncertainty.
    """
    pi_decayed = pi + decay_rate * days_elapsed
    return max(0.0, min(1.0, pi_decayed))
