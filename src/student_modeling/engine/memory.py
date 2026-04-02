from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from student_modeling.engine.kalman import (
    compute_gain,
    compute_residual,
    compute_time_decay,
    update_memory,
    update_perplexity,
)

# Default constants (can be overridden via Settings)
R_POS = 0.5
R_NEG = 1.0
Q_TASK = 0.05
Q_TIME = 0.05
PI_INIT = 1.0
TIME_DECAY_RATE = 0.02


def initialize_knows_edge(concept_emb: np.ndarray) -> dict:
    """Create initial KNOWS edge properties for a new user-concept interaction.

    m_task starts as the concept embedding (normalized).
    Both perplexities start at 1.0 (maximum uncertainty).
    Mastery starts at 0.0.
    """
    norm = np.linalg.norm(concept_emb)
    m_task = concept_emb / norm if norm > 1e-10 else concept_emb.copy()
    return {
        "m_task": m_task,
        "m_time_last": datetime.now(timezone.utc),
        "pi_task": PI_INIT,
        "pi_time": PI_INIT,
        "mastery": 0.0,
        "attempts": 0,
    }


def derive_mastery(
    m_task: np.ndarray, concept_emb: np.ndarray, pi_task: float, pi_time: float
) -> float:
    """Derive mastery from memory state.

    mastery = clip[0,1]((1 - pi_task) * cos(m_task, concept_emb) * (1 - pi_time))

    High only when: task perplexity low, memory aligned with concept, time perplexity low.
    """
    cos_sim = float(np.dot(m_task, concept_emb))
    mastery = (1.0 - pi_task) * cos_sim * (1.0 - pi_time)
    return max(0.0, min(1.0, mastery))


def process_feedback(
    y: int,
    query_emb: np.ndarray,
    concept_emb: np.ndarray,
    m_task: np.ndarray,
    m_time_last: datetime,
    pi_task: float,
    pi_time: float,
    mastery: float,
    attempts: int,
    r_pos: float = R_POS,
    r_neg: float = R_NEG,
    q_task: float = Q_TASK,
    q_time: float = Q_TIME,
    time_decay_rate: float = TIME_DECAY_RATE,
) -> dict:
    """Full GAM-RAG update cycle for a single feedback event.

    1. Apply time decay to pi_time and pi_task
    2. Select observation noise R based on feedback
    3. Compute Kalman gain K
    4. Compute residual e
    5. Update memory vector m_task
    6. Update task perplexity pi_task
    7. Reset time state
    8. Derive new mastery
    9. Increment attempts
    """
    now = datetime.now(timezone.utc)

    # 1. Time decay
    if m_time_last.tzinfo is None:
        m_time_last = m_time_last.replace(tzinfo=timezone.utc)
    days_elapsed = (now - m_time_last).total_seconds() / 86400.0
    pi_task = compute_time_decay(pi_task, days_elapsed, time_decay_rate)
    pi_time = compute_time_decay(pi_time, days_elapsed, time_decay_rate)

    # 2. Observation noise
    R = r_pos if y == 1 else r_neg

    # 3. Kalman gain
    K = compute_gain(pi_task, R)

    # 4. Residual
    e = compute_residual(y, query_emb, m_task)

    # 5. Memory update
    m_task_new = update_memory(m_task, K, e, query_emb)

    # 6. Perplexity update
    pi_task_new = update_perplexity(pi_task, K, q_task)

    # 7. Reset time state (fresh interaction)
    pi_time_new = q_time  # floor value after fresh interaction

    # 8. Derive mastery
    mastery_new = derive_mastery(m_task_new, concept_emb, pi_task_new, pi_time_new)

    # 9. Return updated edge properties
    return {
        "m_task": m_task_new,
        "m_time_last": now,
        "pi_task": pi_task_new,
        "pi_time": pi_time_new,
        "mastery": mastery_new,
        "attempts": attempts + 1,
    }
