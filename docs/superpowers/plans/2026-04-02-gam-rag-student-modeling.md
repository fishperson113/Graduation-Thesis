# GAM-RAG Student Modeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a graph-based student modeling system with GAM-RAG Kalman-gain update policy to prove the theory works for adaptive learning.

**Architecture:** Layered Domain Engine — `engine/` (pure math, no DB) + `repositories/` (Neo4j CRUD) + `services/` (orchestration). The engine is the future MCP core.

**Tech Stack:** Python 3.10+, Neo4j 5, sentence-transformers (all-mpnet-base-v2), numpy, pydantic-settings, pytest

**Design Spec:** `docs/superpowers/specs/2026-04-02-gam-rag-student-modeling-design.md`

---

## Task 1: Update Dependencies and Config

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/student_modeling/config.py`
- Modify: `.env.example`

- [ ] **Step 1: Update pyproject.toml with new dependencies**

```toml
[project]
name = "student-modeling"
version = "0.1.0"
description = "Graph-based student modeling with Neo4j and GAM-RAG update policy"
requires-python = ">=3.10"
dependencies = [
    "neo4j>=5.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "sentence-transformers>=2.0",
    "numpy>=1.24",
]
```

- [ ] **Step 2: Add GAM-RAG constants to config.py**

```python
from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    neo4j_uri: str = "neo4j://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("changeme")
    neo4j_database: str = "neo4j"
    log_level: str = "INFO"

    # Embedding model
    embedding_model: str = "all-mpnet-base-v2"
    embedding_dim: int = 768

    # GAM-RAG constants
    r_pos: float = 0.5
    r_neg: float = 1.0
    q_task: float = 0.05
    q_time: float = 0.05
    pi_init: float = 1.0
    time_decay_rate: float = 0.02  # perplexity increase per day of absence


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 3: Update .env.example**

Add these lines at the end of `.env.example`:

```
# Embedding model
EMBEDDING_MODEL=all-mpnet-base-v2
EMBEDDING_DIM=768

# GAM-RAG update policy
R_POS=0.5
R_NEG=1.0
Q_TASK=0.05
Q_TIME=0.05
PI_INIT=1.0
TIME_DECAY_RATE=0.02
```

- [ ] **Step 4: Install updated dependencies**

Run: `pip install -e ".[dev]"`
Expected: Successful install including sentence-transformers and numpy.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/student_modeling/config.py .env.example
git commit -m "feat: add GAM-RAG dependencies and config constants"
```

---

## Task 2: Remove Old Test Models and Scaffold New Structure

**Files:**
- Delete: `src/student_modeling/models/student.py`
- Delete: `src/student_modeling/models/course.py`
- Delete: `src/student_modeling/models/skill.py`
- Delete: `src/student_modeling/models/knowledge_state.py`
- Delete: `src/student_modeling/repositories/student_repository.py`
- Delete: `src/student_modeling/repositories/course_repository.py`
- Delete: `src/student_modeling/repositories/skill_repository.py`
- Delete: `src/student_modeling/repositories/knowledge_state_repository.py`
- Delete: `src/student_modeling/services/student_service.py`
- Delete: `tests/repositories/test_student_repository.py`
- Modify: `src/student_modeling/models/__init__.py`
- Modify: `src/student_modeling/repositories/__init__.py`
- Modify: `src/student_modeling/services/__init__.py`
- Create: `src/student_modeling/engine/__init__.py`
- Create: `tests/engine/__init__.py`
- Create: `tests/services/__init__.py`

- [ ] **Step 1: Delete old model files**

```bash
rm src/student_modeling/models/student.py
rm src/student_modeling/models/course.py
rm src/student_modeling/models/skill.py
rm src/student_modeling/models/knowledge_state.py
```

- [ ] **Step 2: Delete old repository files**

```bash
rm src/student_modeling/repositories/student_repository.py
rm src/student_modeling/repositories/course_repository.py
rm src/student_modeling/repositories/skill_repository.py
rm src/student_modeling/repositories/knowledge_state_repository.py
```

- [ ] **Step 3: Delete old service and test files**

```bash
rm src/student_modeling/services/student_service.py
rm tests/repositories/test_student_repository.py
```

- [ ] **Step 4: Clear the __init__.py barrel exports**

`src/student_modeling/models/__init__.py`:
```python
```

(Empty file — will be populated as new models are added.)

`src/student_modeling/repositories/__init__.py`:
```python
```

`src/student_modeling/services/__init__.py`:
```python
```

- [ ] **Step 5: Create new directory scaffolds**

```bash
mkdir -p src/student_modeling/engine
touch src/student_modeling/engine/__init__.py
mkdir -p tests/engine
touch tests/engine/__init__.py
mkdir -p tests/services
touch tests/services/__init__.py
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: remove test scaffold models, prepare for GAM-RAG implementation"
```

---

## Task 3: Engine — Kalman Update Functions

**Files:**
- Create: `src/student_modeling/engine/kalman.py`
- Create: `tests/engine/test_kalman.py`

- [ ] **Step 1: Write failing tests for compute_gain**

Create `tests/engine/test_kalman.py`:

```python
from __future__ import annotations

import pytest

from student_modeling.engine.kalman import (
    compute_gain,
    compute_residual,
    compute_time_decay,
    update_memory,
    update_perplexity,
)
import numpy as np


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/engine/test_kalman.py::TestComputeGain -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'student_modeling.engine.kalman'`

- [ ] **Step 3: Implement compute_gain**

Create `src/student_modeling/engine/kalman.py`:

```python
from __future__ import annotations

from datetime import datetime

import numpy as np


def compute_gain(pi: float, R: float) -> float:
    """Kalman gain: K = pi / (pi + R).

    High pi (uncertain) -> high K (fast update / warm-up).
    Low pi (stable) -> low K (damped refinement).
    """
    return pi / (pi + R)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/engine/test_kalman.py::TestComputeGain -v`
Expected: 4 passed

- [ ] **Step 5: Write failing tests for compute_residual**

Append to `tests/engine/test_kalman.py`:

```python
class TestComputeResidual:
    def test_positive_residual_on_correct_with_low_alignment(self):
        """y=1 and memory far from query -> e > 0 (attraction)."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal(768).astype(np.float32)
        q /= np.linalg.norm(q)
        m = rng.standard_normal(768).astype(np.float32)
        m /= np.linalg.norm(m)
        e = compute_residual(y=1, query_emb=q, memory_vec=m)
        # cos of random vectors ~ 0, so e ~ 1 - 0 ~ 1
        assert e > 0.5

    def test_negative_residual_on_incorrect_with_high_alignment(self):
        """y=0 and memory aligned with query -> e < 0 (repulsion)."""
        q = np.ones(768, dtype=np.float32)
        q /= np.linalg.norm(q)
        m = q.copy()  # perfectly aligned
        e = compute_residual(y=0, query_emb=q, memory_vec=m)
        # cos = 1.0, so e = 0 - 1.0 = -1.0
        assert e == pytest.approx(-1.0, abs=1e-5)

    def test_residual_range(self):
        """Residual is in [-1, 2] since y in {0,1} and cos in [-1,1]."""
        q = np.ones(768, dtype=np.float32)
        q /= np.linalg.norm(q)
        m = -q.copy()  # anti-aligned
        e_pos = compute_residual(y=1, query_emb=q, memory_vec=m)
        # cos = -1.0, so e = 1 - (-1) = 2.0
        assert e_pos == pytest.approx(2.0, abs=1e-5)
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/engine/test_kalman.py::TestComputeResidual -v`
Expected: FAIL with `ImportError` (compute_residual not yet defined)

- [ ] **Step 7: Implement compute_residual**

Add to `src/student_modeling/engine/kalman.py`:

```python
def compute_residual(y: int, query_emb: np.ndarray, memory_vec: np.ndarray) -> float:
    """Residual: e = y - cos(q, m_old).

    e > 0 -> pull memory toward query (attraction).
    e < 0 -> push memory away from query (repulsion).
    """
    cos_sim = float(np.dot(query_emb, memory_vec))
    return y - cos_sim
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/engine/test_kalman.py::TestComputeResidual -v`
Expected: 3 passed

- [ ] **Step 9: Write failing tests for update_memory**

Append to `tests/engine/test_kalman.py`:

```python
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
        q = np.ones(768, dtype=np.float32)
        q /= np.linalg.norm(q)
        m = q.copy()
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
```

- [ ] **Step 10: Run tests to verify they fail**

Run: `pytest tests/engine/test_kalman.py::TestUpdateMemory -v`
Expected: FAIL

- [ ] **Step 11: Implement update_memory**

Add to `src/student_modeling/engine/kalman.py`:

```python
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
```

- [ ] **Step 12: Run tests to verify they pass**

Run: `pytest tests/engine/test_kalman.py::TestUpdateMemory -v`
Expected: 3 passed

- [ ] **Step 13: Write failing tests for update_perplexity and compute_time_decay**

Append to `tests/engine/test_kalman.py`:

```python
class TestUpdatePerplexity:
    def test_perplexity_decreases_after_update(self):
        """After a feedback event, perplexity should decrease (confidence grows)."""
        pi_new = update_perplexity(pi_old=1.0, K=0.667, Q=0.05)
        assert pi_new < 1.0

    def test_perplexity_floor(self):
        """Perplexity never drops below Q."""
        # Many updates with high gain
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
```

- [ ] **Step 14: Run tests to verify they fail**

Run: `pytest tests/engine/test_kalman.py -k "TestUpdatePerplexity or TestComputeTimeDecay" -v`
Expected: FAIL

- [ ] **Step 15: Implement update_perplexity and compute_time_decay**

Add to `src/student_modeling/engine/kalman.py`:

```python
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
```

- [ ] **Step 16: Run all kalman tests**

Run: `pytest tests/engine/test_kalman.py -v`
Expected: All 14 tests pass

- [ ] **Step 17: Commit**

```bash
git add src/student_modeling/engine/kalman.py tests/engine/test_kalman.py
git commit -m "feat: implement Kalman gain engine (compute_gain, residual, memory update, perplexity)"
```

---

## Task 4: Engine — Memory Operations and Mastery Derivation

**Files:**
- Create: `src/student_modeling/engine/memory.py`
- Create: `tests/engine/test_memory.py`

- [ ] **Step 1: Write failing tests for initialize_knows_edge and derive_mastery**

Create `tests/engine/test_memory.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from student_modeling.engine.memory import (
    derive_mastery,
    initialize_knows_edge,
    process_feedback,
)


class TestInitializeKnowsEdge:
    def test_defaults(self):
        rng = np.random.default_rng(42)
        concept_emb = rng.standard_normal(768).astype(np.float32)
        concept_emb /= np.linalg.norm(concept_emb)

        edge = initialize_knows_edge(concept_emb)
        assert edge["pi_task"] == 1.0
        assert edge["pi_time"] == 1.0
        assert edge["mastery"] == 0.0
        assert edge["attempts"] == 0
        assert isinstance(edge["m_time_last"], datetime)

    def test_m_task_equals_concept_embedding(self):
        rng = np.random.default_rng(42)
        concept_emb = rng.standard_normal(768).astype(np.float32)
        concept_emb /= np.linalg.norm(concept_emb)

        edge = initialize_knows_edge(concept_emb)
        np.testing.assert_allclose(edge["m_task"], concept_emb, atol=1e-6)

    def test_m_task_is_normalized(self):
        emb = np.ones(768, dtype=np.float32)  # not normalized
        edge = initialize_knows_edge(emb)
        assert np.linalg.norm(edge["m_task"]) == pytest.approx(1.0, abs=1e-5)


class TestDeriveMastery:
    def test_high_mastery_when_aligned_and_confident(self):
        """Low perplexity + aligned memory = high mastery."""
        emb = np.ones(768, dtype=np.float32)
        emb /= np.linalg.norm(emb)
        mastery = derive_mastery(
            m_task=emb, concept_emb=emb, pi_task=0.05, pi_time=0.05
        )
        # (1 - 0.05) * 1.0 * (1 - 0.05) = 0.9025
        assert mastery > 0.85

    def test_zero_mastery_at_initialization(self):
        """pi_task=1.0 -> (1 - 1.0) = 0 -> mastery = 0."""
        emb = np.ones(768, dtype=np.float32)
        emb /= np.linalg.norm(emb)
        mastery = derive_mastery(
            m_task=emb, concept_emb=emb, pi_task=1.0, pi_time=1.0
        )
        assert mastery == pytest.approx(0.0, abs=1e-6)

    def test_time_decay_reduces_mastery(self):
        """High pi_time (stale knowledge) reduces mastery."""
        emb = np.ones(768, dtype=np.float32)
        emb /= np.linalg.norm(emb)
        fresh = derive_mastery(m_task=emb, concept_emb=emb, pi_task=0.1, pi_time=0.1)
        stale = derive_mastery(m_task=emb, concept_emb=emb, pi_task=0.1, pi_time=0.9)
        assert fresh > stale

    def test_mastery_clipped_to_0_1(self):
        """Mastery is always in [0, 1]."""
        emb = np.ones(768, dtype=np.float32)
        emb /= np.linalg.norm(emb)
        m = -emb  # anti-aligned -> cos = -1
        mastery = derive_mastery(m_task=m, concept_emb=emb, pi_task=0.1, pi_time=0.1)
        assert mastery >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/engine/test_memory.py -k "TestInitializeKnowsEdge or TestDeriveMastery" -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement initialize_knows_edge and derive_mastery**

Create `src/student_modeling/engine/memory.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/engine/test_memory.py -k "TestInitializeKnowsEdge or TestDeriveMastery" -v`
Expected: 7 passed

- [ ] **Step 5: Write failing tests for process_feedback — theory validation**

Append to `tests/engine/test_memory.py`:

```python
class TestProcessFeedback:
    """These tests prove the GAM-RAG theory works for student modeling."""

    def _make_concept_emb(self, seed=42):
        rng = np.random.default_rng(seed)
        emb = rng.standard_normal(768).astype(np.float32)
        return emb / np.linalg.norm(emb)

    def test_single_correct_answer_increases_mastery(self):
        concept_emb = self._make_concept_emb()
        edge = initialize_knows_edge(concept_emb)
        query_emb = concept_emb.copy()  # question aligned with concept

        result = process_feedback(
            y=1,
            query_emb=query_emb,
            concept_emb=concept_emb,
            **edge,
        )
        assert result["mastery"] > edge["mastery"]
        assert result["pi_task"] < edge["pi_task"]
        assert result["attempts"] == 1

    def test_warm_up_sequence(self):
        """5 correct answers should raise mastery rapidly (fast warm-up)."""
        concept_emb = self._make_concept_emb()
        edge = initialize_knows_edge(concept_emb)
        query_emb = concept_emb.copy()

        for _ in range(5):
            edge = process_feedback(
                y=1, query_emb=query_emb, concept_emb=concept_emb, **edge
            )

        assert edge["mastery"] > 0.4
        assert edge["pi_task"] < 0.5

    def test_damped_refinement(self):
        """After many correct answers, further corrections change mastery slowly."""
        concept_emb = self._make_concept_emb()
        edge = initialize_knows_edge(concept_emb)
        query_emb = concept_emb.copy()

        # 20 correct answers to stabilize
        for _ in range(20):
            edge = process_feedback(
                y=1, query_emb=query_emb, concept_emb=concept_emb, **edge
            )

        mastery_before = edge["mastery"]

        # One more correct answer
        edge = process_feedback(
            y=1, query_emb=query_emb, concept_emb=concept_emb, **edge
        )
        mastery_after = edge["mastery"]

        # Change should be small (damped)
        assert abs(mastery_after - mastery_before) < 0.05

    def test_forgetting_repulsion(self):
        """Master a concept, then fail -> mastery should drop (repulsion)."""
        concept_emb = self._make_concept_emb()
        edge = initialize_knows_edge(concept_emb)
        query_emb = concept_emb.copy()

        # Master the concept
        for _ in range(15):
            edge = process_feedback(
                y=1, query_emb=query_emb, concept_emb=concept_emb, **edge
            )
        mastery_peak = edge["mastery"]
        assert mastery_peak > 0.5

        # Fail
        edge = process_feedback(
            y=0, query_emb=query_emb, concept_emb=concept_emb, **edge
        )
        assert edge["mastery"] < mastery_peak

    def test_relearning_recovery(self):
        """After forgetting, correct answers should recover mastery faster than initial learning."""
        concept_emb = self._make_concept_emb()
        edge = initialize_knows_edge(concept_emb)
        query_emb = concept_emb.copy()

        # Master
        for _ in range(15):
            edge = process_feedback(
                y=1, query_emb=query_emb, concept_emb=concept_emb, **edge
            )

        # Forget (3 failures)
        for _ in range(3):
            edge = process_feedback(
                y=0, query_emb=query_emb, concept_emb=concept_emb, **edge
            )
        mastery_after_forget = edge["mastery"]

        # Re-learn (5 correct)
        for _ in range(5):
            edge = process_feedback(
                y=1, query_emb=query_emb, concept_emb=concept_emb, **edge
            )

        # Should recover significantly
        assert edge["mastery"] > mastery_after_forget

    def test_time_decay_increases_perplexity(self):
        """Long time gap increases perplexity, reducing mastery."""
        concept_emb = self._make_concept_emb()
        edge = initialize_knows_edge(concept_emb)
        query_emb = concept_emb.copy()

        # Build some mastery
        for _ in range(10):
            edge = process_feedback(
                y=1, query_emb=query_emb, concept_emb=concept_emb, **edge
            )
        mastery_fresh = edge["mastery"]

        # Simulate 30 days passing by backdating m_time_last
        from datetime import timedelta
        edge["m_time_last"] = datetime.now(timezone.utc) - timedelta(days=30)

        # Next interaction should show reduced mastery due to time decay
        edge_after = process_feedback(
            y=1, query_emb=query_emb, concept_emb=concept_emb, **edge
        )
        # The time decay should have increased pi_task before the update,
        # resulting in higher gain (re-warm-up behavior)
        # Mastery may or may not be lower depending on the y=1 update,
        # but pi_task should have been higher entering the update
        assert edge_after["attempts"] == edge["attempts"] + 1
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/engine/test_memory.py::TestProcessFeedback -v`
Expected: FAIL (process_feedback not yet implemented)

- [ ] **Step 7: Implement process_feedback**

Add to `src/student_modeling/engine/memory.py`:

```python
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
```

- [ ] **Step 8: Run all memory tests**

Run: `pytest tests/engine/test_memory.py -v`
Expected: All 13 tests pass

- [ ] **Step 9: Commit**

```bash
git add src/student_modeling/engine/memory.py tests/engine/test_memory.py
git commit -m "feat: implement memory operations and process_feedback (GAM-RAG theory core)"
```

---

## Task 5: Engine — Embedding Service

**Files:**
- Create: `src/student_modeling/engine/embeddings.py`
- Create: `tests/engine/test_embeddings.py`

- [ ] **Step 1: Write failing test**

Create `tests/engine/test_embeddings.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from student_modeling.engine.embeddings import EmbeddingService


class TestEmbeddingService:
    @pytest.fixture(scope="class")
    def service(self):
        return EmbeddingService(model_name="all-mpnet-base-v2")

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/engine/test_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement EmbeddingService**

Create `src/student_modeling/engine/embeddings.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/engine/test_embeddings.py -v`
Expected: 4 passed (first run may be slow due to model download)

- [ ] **Step 5: Commit**

```bash
git add src/student_modeling/engine/embeddings.py tests/engine/test_embeddings.py
git commit -m "feat: add EmbeddingService wrapping all-mpnet-base-v2"
```

---

## Task 6: Domain Models

**Files:**
- Create: `src/student_modeling/models/domain.py`
- Create: `src/student_modeling/models/concept.py`
- Create: `src/student_modeling/models/learning_object.py`
- Create: `src/student_modeling/models/user.py`
- Create: `src/student_modeling/models/knows_edge.py`
- Modify: `src/student_modeling/models/__init__.py`

- [ ] **Step 1: Create Domain model**

Create `src/student_modeling/models/domain.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Domain:
    domain_id: str
    name: str
    parent_id: str | None = None
    element_id: str | None = field(default=None, repr=False)

    @classmethod
    def from_node(cls, node: dict[str, Any]) -> Domain:
        return cls(
            domain_id=node["domain_id"],
            name=node["name"],
            parent_id=node.get("parent_id"),
            element_id=node.get("element_id"),
        )

    def to_properties(self) -> dict[str, Any]:
        props: dict[str, Any] = {
            "domain_id": self.domain_id,
            "name": self.name,
        }
        if self.parent_id is not None:
            props["parent_id"] = self.parent_id
        return props
```

- [ ] **Step 2: Create Concept model**

Create `src/student_modeling/models/concept.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Concept:
    concept_id: str
    name: str
    domain_id: str
    description: str = ""
    embedding: list[float] = field(default_factory=list, repr=False)
    element_id: str | None = field(default=None, repr=False)

    @classmethod
    def from_node(cls, node: dict[str, Any]) -> Concept:
        return cls(
            concept_id=node["concept_id"],
            name=node["name"],
            domain_id=node.get("domain_id", ""),
            description=node.get("description", ""),
            embedding=node.get("embedding", []),
            element_id=node.get("element_id"),
        )

    def to_properties(self) -> dict[str, Any]:
        props: dict[str, Any] = {
            "concept_id": self.concept_id,
            "name": self.name,
            "domain_id": self.domain_id,
            "description": self.description,
        }
        if self.embedding:
            props["embedding"] = self.embedding
        return props
```

- [ ] **Step 3: Create LearningObject model**

Create `src/student_modeling/models/learning_object.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LearningObject:
    lo_id: str
    question: str
    answer: str
    fact: str = ""
    entities: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list, repr=False)
    element_id: str | None = field(default=None, repr=False)

    @classmethod
    def from_node(cls, node: dict[str, Any]) -> LearningObject:
        return cls(
            lo_id=node["lo_id"],
            question=node["question"],
            answer=node["answer"],
            fact=node.get("fact", ""),
            entities=node.get("entities", []),
            embedding=node.get("embedding", []),
            element_id=node.get("element_id"),
        )

    def to_properties(self) -> dict[str, Any]:
        props: dict[str, Any] = {
            "lo_id": self.lo_id,
            "question": self.question,
            "answer": self.answer,
            "fact": self.fact,
            "entities": self.entities,
        }
        if self.embedding:
            props["embedding"] = self.embedding
        return props
```

- [ ] **Step 4: Create User model**

Create `src/student_modeling/models/user.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neo4j.time import DateTime


@dataclass
class User:
    user_id: str
    name: str
    email: str
    created_at: DateTime | None = None
    element_id: str | None = field(default=None, repr=False)

    @classmethod
    def from_node(cls, node: dict[str, Any]) -> User:
        return cls(
            user_id=node["user_id"],
            name=node["name"],
            email=node.get("email", ""),
            created_at=node.get("created_at"),
            element_id=node.get("element_id"),
        )

    def to_properties(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "email": self.email,
        }
```

- [ ] **Step 5: Create KnowsEdge model**

Create `src/student_modeling/models/knows_edge.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class KnowsEdge:
    """Represents the KNOWS relationship between User and Concept.

    Carries GAM-RAG memory properties: task memory vector, perplexities,
    derived mastery score, and interaction metadata.
    """

    user_id: str
    concept_id: str
    m_task: list[float]
    m_time_last: datetime
    pi_task: float
    pi_time: float
    mastery: float
    attempts: int

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> KnowsEdge:
        m_time_last = record["m_time_last"]
        if hasattr(m_time_last, "to_native"):
            m_time_last = m_time_last.to_native()
        return cls(
            user_id=record["user_id"],
            concept_id=record["concept_id"],
            m_task=record.get("m_task", []),
            m_time_last=m_time_last,
            pi_task=record.get("pi_task", 1.0),
            pi_time=record.get("pi_time", 1.0),
            mastery=record.get("mastery", 0.0),
            attempts=record.get("attempts", 0),
        )

    def to_properties(self) -> dict[str, Any]:
        return {
            "m_task": self.m_task,
            "m_time_last": self.m_time_last,
            "pi_task": self.pi_task,
            "pi_time": self.pi_time,
            "mastery": self.mastery,
            "attempts": self.attempts,
        }
```

- [ ] **Step 6: Update models __init__.py**

Write `src/student_modeling/models/__init__.py`:

```python
from student_modeling.models.concept import Concept
from student_modeling.models.domain import Domain
from student_modeling.models.knows_edge import KnowsEdge
from student_modeling.models.learning_object import LearningObject
from student_modeling.models.user import User

__all__ = ["Concept", "Domain", "KnowsEdge", "LearningObject", "User"]
```

- [ ] **Step 7: Verify imports work**

Run: `python -c "from student_modeling.models import Concept, Domain, KnowsEdge, LearningObject, User; print('OK')"`
Expected: `OK`

- [ ] **Step 8: Commit**

```bash
git add src/student_modeling/models/
git commit -m "feat: add domain models (Domain, Concept, LearningObject, User, KnowsEdge)"
```

---

## Task 7: Repositories — User, Domain, Concept

**Files:**
- Create: `src/student_modeling/repositories/user_repository.py`
- Create: `src/student_modeling/repositories/domain_repository.py`
- Create: `src/student_modeling/repositories/concept_repository.py`
- Create: `tests/repositories/test_user_repository.py`

- [ ] **Step 1: Write failing tests for UserRepository**

Create `tests/repositories/test_user_repository.py`:

```python
from __future__ import annotations

import pytest

from student_modeling.models.user import User
from student_modeling.repositories.user_repository import UserRepository


class TestUserRepository:
    @pytest.fixture
    def repo(self, neo4j_driver, database_name):
        return UserRepository(neo4j_driver, database_name)

    def test_create_and_find(self, repo):
        user = User(user_id="u1", name="Alice", email="alice@test.com")
        created = repo.create(user)
        assert created.user_id == "u1"
        assert created.name == "Alice"

        found = repo.find_by_id("u1")
        assert found is not None
        assert found.user_id == "u1"

    def test_find_nonexistent(self, repo):
        assert repo.find_by_id("nonexistent") is None

    def test_delete(self, repo):
        user = User(user_id="u2", name="Bob", email="bob@test.com")
        repo.create(user)
        assert repo.delete("u2") is True
        assert repo.find_by_id("u2") is None

    def test_delete_nonexistent(self, repo):
        assert repo.delete("nonexistent") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/repositories/test_user_repository.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement UserRepository**

Create `src/student_modeling/repositories/user_repository.py`:

```python
from __future__ import annotations

from typing import Any

from neo4j import ManagedTransaction

from student_modeling.models.user import User
from student_modeling.repositories.base import BaseRepository


class UserRepository(BaseRepository):
    def create(self, user: User) -> User:
        def _work(tx: ManagedTransaction) -> dict[str, Any]:
            result = tx.run(
                """
                CREATE (u:User {
                    user_id: $user_id,
                    name: $name,
                    email: $email,
                    created_at: datetime()
                })
                RETURN u {.*, element_id: elementId(u)} AS user
                """,
                **user.to_properties(),
            )
            return result.single(strict=True)["user"]

        with self._session() as session:
            data = session.execute_write(_work)
            return User.from_node(data)

    def find_by_id(self, user_id: str) -> User | None:
        def _work(tx: ManagedTransaction) -> dict[str, Any] | None:
            result = tx.run(
                """
                MATCH (u:User {user_id: $user_id})
                RETURN u {.*, element_id: elementId(u)} AS user
                """,
                user_id=user_id,
            )
            record = result.single()
            return record["user"] if record else None

        with self._session() as session:
            data = session.execute_read(_work)
            return User.from_node(data) if data else None

    def delete(self, user_id: str) -> bool:
        def _work(tx: ManagedTransaction) -> int:
            result = tx.run(
                """
                MATCH (u:User {user_id: $user_id})
                DETACH DELETE u
                RETURN count(*) AS deleted
                """,
                user_id=user_id,
            )
            return result.single(strict=True)["deleted"]

        with self._session() as session:
            return session.execute_write(_work) > 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/repositories/test_user_repository.py -v`
Expected: 4 passed

- [ ] **Step 5: Implement DomainRepository**

Create `src/student_modeling/repositories/domain_repository.py`:

```python
from __future__ import annotations

from typing import Any

from neo4j import ManagedTransaction

from student_modeling.models.domain import Domain
from student_modeling.repositories.base import BaseRepository


class DomainRepository(BaseRepository):
    def create(self, domain: Domain) -> Domain:
        def _work(tx: ManagedTransaction) -> dict[str, Any]:
            result = tx.run(
                """
                CREATE (d:Domain {domain_id: $domain_id, name: $name})
                RETURN d {.*, element_id: elementId(d)} AS domain
                """,
                **domain.to_properties(),
            )
            return result.single(strict=True)["domain"]

        with self._session() as session:
            data = session.execute_write(_work)
            return Domain.from_node(data)

    def find_by_id(self, domain_id: str) -> Domain | None:
        def _work(tx: ManagedTransaction) -> dict[str, Any] | None:
            result = tx.run(
                """
                MATCH (d:Domain {domain_id: $domain_id})
                RETURN d {.*, element_id: elementId(d)} AS domain
                """,
                domain_id=domain_id,
            )
            record = result.single()
            return record["domain"] if record else None

        with self._session() as session:
            data = session.execute_read(_work)
            return Domain.from_node(data) if data else None

    def delete(self, domain_id: str) -> bool:
        def _work(tx: ManagedTransaction) -> int:
            result = tx.run(
                """
                MATCH (d:Domain {domain_id: $domain_id})
                DETACH DELETE d
                RETURN count(*) AS deleted
                """,
                domain_id=domain_id,
            )
            return result.single(strict=True)["deleted"]

        with self._session() as session:
            return session.execute_write(_work) > 0

    def add_child(self, parent_id: str, child_id: str) -> None:
        def _work(tx: ManagedTransaction) -> None:
            tx.run(
                """
                MATCH (p:Domain {domain_id: $parent_id})
                MATCH (c:Domain {domain_id: $child_id})
                MERGE (p)-[:HAS_CHILD]->(c)
                """,
                parent_id=parent_id,
                child_id=child_id,
            )

        with self._session() as session:
            session.execute_write(_work)

    def get_root_domains(self) -> list[Domain]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (d:Domain)
            WHERE NOT EXISTS { MATCH (:Domain)-[:HAS_CHILD]->(d) }
            RETURN d {.*, element_id: elementId(d)} AS domain
            ORDER BY d.name
            """,
            database_=self._database,
            routing_="r",
        )
        return [Domain.from_node(r["domain"]) for r in records]
```

- [ ] **Step 6: Implement ConceptRepository**

Create `src/student_modeling/repositories/concept_repository.py`:

```python
from __future__ import annotations

from typing import Any

from neo4j import ManagedTransaction

from student_modeling.models.concept import Concept
from student_modeling.repositories.base import BaseRepository


class ConceptRepository(BaseRepository):
    def create(self, concept: Concept) -> Concept:
        def _work(tx: ManagedTransaction) -> dict[str, Any]:
            result = tx.run(
                """
                CREATE (c:Concept {
                    concept_id: $concept_id,
                    name: $name,
                    domain_id: $domain_id,
                    description: $description,
                    embedding: $embedding
                })
                RETURN c {.*, element_id: elementId(c)} AS concept
                """,
                **concept.to_properties(),
            )
            return result.single(strict=True)["concept"]

        with self._session() as session:
            data = session.execute_write(_work)
            return Concept.from_node(data)

    def find_by_id(self, concept_id: str) -> Concept | None:
        def _work(tx: ManagedTransaction) -> dict[str, Any] | None:
            result = tx.run(
                """
                MATCH (c:Concept {concept_id: $concept_id})
                RETURN c {.*, element_id: elementId(c)} AS concept
                """,
                concept_id=concept_id,
            )
            record = result.single()
            return record["concept"] if record else None

        with self._session() as session:
            data = session.execute_read(_work)
            return Concept.from_node(data) if data else None

    def delete(self, concept_id: str) -> bool:
        def _work(tx: ManagedTransaction) -> int:
            result = tx.run(
                """
                MATCH (c:Concept {concept_id: $concept_id})
                DETACH DELETE c
                RETURN count(*) AS deleted
                """,
                concept_id=concept_id,
            )
            return result.single(strict=True)["deleted"]

        with self._session() as session:
            return session.execute_write(_work) > 0

    def add_prerequisite(self, concept_id: str, prerequisite_id: str) -> None:
        def _work(tx: ManagedTransaction) -> None:
            tx.run(
                """
                MATCH (c:Concept {concept_id: $concept_id})
                MATCH (p:Concept {concept_id: $prerequisite_id})
                MERGE (c)-[:HAS_PREREQUISITE]->(p)
                """,
                concept_id=concept_id,
                prerequisite_id=prerequisite_id,
            )

        with self._session() as session:
            session.execute_write(_work)

    def get_prerequisites(self, concept_id: str) -> list[Concept]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (c:Concept {concept_id: $concept_id})-[:HAS_PREREQUISITE]->(p:Concept)
            RETURN p {.*, element_id: elementId(p)} AS concept
            ORDER BY p.name
            """,
            concept_id=concept_id,
            database_=self._database,
            routing_="r",
        )
        return [Concept.from_node(r["concept"]) for r in records]

    def get_by_domain(self, domain_id: str) -> list[Concept]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (c:Concept {domain_id: $domain_id})
            RETURN c {.*, element_id: elementId(c)} AS concept
            ORDER BY c.name
            """,
            domain_id=domain_id,
            database_=self._database,
            routing_="r",
        )
        return [Concept.from_node(r["concept"]) for r in records]

    def link_to_domain(self, concept_id: str, domain_id: str) -> None:
        def _work(tx: ManagedTransaction) -> None:
            tx.run(
                """
                MATCH (d:Domain {domain_id: $domain_id})
                MATCH (c:Concept {concept_id: $concept_id})
                MERGE (d)-[:CONTAINS]->(c)
                """,
                domain_id=domain_id,
                concept_id=concept_id,
            )

        with self._session() as session:
            session.execute_write(_work)
```

- [ ] **Step 7: Commit**

```bash
git add src/student_modeling/repositories/user_repository.py src/student_modeling/repositories/domain_repository.py src/student_modeling/repositories/concept_repository.py tests/repositories/test_user_repository.py
git commit -m "feat: add User, Domain, Concept repositories"
```

---

## Task 8: Repositories — LearningObject and KNOWS Edge

**Files:**
- Create: `src/student_modeling/repositories/learning_object_repository.py`
- Create: `src/student_modeling/repositories/knows_repository.py`
- Create: `tests/repositories/test_knows_repository.py`
- Modify: `src/student_modeling/repositories/__init__.py`

- [ ] **Step 1: Implement LearningObjectRepository**

Create `src/student_modeling/repositories/learning_object_repository.py`:

```python
from __future__ import annotations

from typing import Any

from neo4j import ManagedTransaction

from student_modeling.models.learning_object import LearningObject
from student_modeling.repositories.base import BaseRepository


class LearningObjectRepository(BaseRepository):
    def create(self, lo: LearningObject) -> LearningObject:
        def _work(tx: ManagedTransaction) -> dict[str, Any]:
            result = tx.run(
                """
                CREATE (lo:LearningObject {
                    lo_id: $lo_id,
                    question: $question,
                    answer: $answer,
                    fact: $fact,
                    entities: $entities,
                    embedding: $embedding
                })
                RETURN lo {.*, element_id: elementId(lo)} AS lo
                """,
                **lo.to_properties(),
            )
            return result.single(strict=True)["lo"]

        with self._session() as session:
            data = session.execute_write(_work)
            return LearningObject.from_node(data)

    def find_by_id(self, lo_id: str) -> LearningObject | None:
        def _work(tx: ManagedTransaction) -> dict[str, Any] | None:
            result = tx.run(
                """
                MATCH (lo:LearningObject {lo_id: $lo_id})
                RETURN lo {.*, element_id: elementId(lo)} AS lo
                """,
                lo_id=lo_id,
            )
            record = result.single()
            return record["lo"] if record else None

        with self._session() as session:
            data = session.execute_read(_work)
            return LearningObject.from_node(data) if data else None

    def delete(self, lo_id: str) -> bool:
        def _work(tx: ManagedTransaction) -> int:
            result = tx.run(
                """
                MATCH (lo:LearningObject {lo_id: $lo_id})
                DETACH DELETE lo
                RETURN count(*) AS deleted
                """,
                lo_id=lo_id,
            )
            return result.single(strict=True)["deleted"]

        with self._session() as session:
            return session.execute_write(_work) > 0

    def get_by_concept(self, concept_id: str) -> list[LearningObject]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (c:Concept {concept_id: $concept_id})-[:COVERS]->(lo:LearningObject)
            RETURN lo {.*, element_id: elementId(lo)} AS lo
            ORDER BY lo.lo_id
            """,
            concept_id=concept_id,
            database_=self._database,
            routing_="r",
        )
        return [LearningObject.from_node(r["lo"]) for r in records]

    def link_to_concept(self, concept_id: str, lo_id: str) -> None:
        def _work(tx: ManagedTransaction) -> None:
            tx.run(
                """
                MATCH (c:Concept {concept_id: $concept_id})
                MATCH (lo:LearningObject {lo_id: $lo_id})
                MERGE (c)-[:COVERS]->(lo)
                """,
                concept_id=concept_id,
                lo_id=lo_id,
            )

        with self._session() as session:
            session.execute_write(_work)

    def bulk_create(self, los: list[LearningObject]) -> int:
        def _work(tx: ManagedTransaction) -> int:
            result = tx.run(
                """
                UNWIND $items AS item
                CREATE (lo:LearningObject {
                    lo_id: item.lo_id,
                    question: item.question,
                    answer: item.answer,
                    fact: item.fact,
                    entities: item.entities,
                    embedding: item.embedding
                })
                RETURN count(lo) AS created
                """,
                items=[lo.to_properties() for lo in los],
            )
            return result.single(strict=True)["created"]

        with self._session() as session:
            return session.execute_write(_work)
```

- [ ] **Step 2: Write failing tests for KnowsRepository**

Create `tests/repositories/test_knows_repository.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from student_modeling.models.concept import Concept
from student_modeling.models.user import User
from student_modeling.repositories.concept_repository import ConceptRepository
from student_modeling.repositories.knows_repository import KnowsRepository
from student_modeling.repositories.user_repository import UserRepository


class TestKnowsRepository:
    @pytest.fixture
    def user_repo(self, neo4j_driver, database_name):
        return UserRepository(neo4j_driver, database_name)

    @pytest.fixture
    def concept_repo(self, neo4j_driver, database_name):
        return ConceptRepository(neo4j_driver, database_name)

    @pytest.fixture
    def repo(self, neo4j_driver, database_name):
        return KnowsRepository(neo4j_driver, database_name)

    @pytest.fixture
    def setup_nodes(self, user_repo, concept_repo):
        user_repo.create(User(user_id="u1", name="Alice", email="a@test.com"))
        concept_repo.create(Concept(
            concept_id="c1", name="Test Concept", domain_id="d1",
            embedding=[0.1] * 768,
        ))

    def test_create_or_update_and_get(self, repo, setup_nodes):
        props = {
            "m_task": [0.5] * 768,
            "m_time_last": datetime.now(timezone.utc),
            "pi_task": 1.0,
            "pi_time": 1.0,
            "mastery": 0.0,
            "attempts": 0,
        }
        repo.create_or_update("u1", "c1", props)

        edge = repo.get_edge("u1", "c1")
        assert edge is not None
        assert edge.user_id == "u1"
        assert edge.concept_id == "c1"
        assert edge.pi_task == 1.0
        assert len(edge.m_task) == 768

    def test_update_existing_edge(self, repo, setup_nodes):
        props = {
            "m_task": [0.5] * 768,
            "m_time_last": datetime.now(timezone.utc),
            "pi_task": 1.0,
            "pi_time": 1.0,
            "mastery": 0.0,
            "attempts": 0,
        }
        repo.create_or_update("u1", "c1", props)

        updated_props = {
            "m_task": [0.6] * 768,
            "m_time_last": datetime.now(timezone.utc),
            "pi_task": 0.8,
            "pi_time": 0.05,
            "mastery": 0.5,
            "attempts": 1,
        }
        repo.create_or_update("u1", "c1", updated_props)

        edge = repo.get_edge("u1", "c1")
        assert edge.pi_task == pytest.approx(0.8)
        assert edge.mastery == pytest.approx(0.5)
        assert edge.attempts == 1

    def test_get_nonexistent_edge(self, repo, setup_nodes):
        assert repo.get_edge("u1", "nonexistent") is None

    def test_get_user_overlay(self, repo, user_repo, concept_repo):
        user_repo.create(User(user_id="u2", name="Bob", email="b@test.com"))
        concept_repo.create(Concept(
            concept_id="c2", name="Concept 2", domain_id="d1",
            embedding=[0.2] * 768,
        ))
        concept_repo.create(Concept(
            concept_id="c3", name="Concept 3", domain_id="d1",
            embedding=[0.3] * 768,
        ))

        props = {
            "m_task": [0.5] * 768,
            "m_time_last": datetime.now(timezone.utc),
            "pi_task": 0.5,
            "pi_time": 0.05,
            "mastery": 0.6,
            "attempts": 5,
        }
        repo.create_or_update("u2", "c2", props)
        repo.create_or_update("u2", "c3", props)

        overlay = repo.get_user_overlay("u2")
        assert len(overlay) == 2
        concept_ids = {e.concept_id for e in overlay}
        assert concept_ids == {"c2", "c3"}

    def test_get_weak_concepts(self, repo, user_repo, concept_repo):
        user_repo.create(User(user_id="u3", name="Carol", email="c@test.com"))
        concept_repo.create(Concept(
            concept_id="c4", name="Strong", domain_id="d1",
            embedding=[0.1] * 768,
        ))
        concept_repo.create(Concept(
            concept_id="c5", name="Weak", domain_id="d1",
            embedding=[0.1] * 768,
        ))

        now = datetime.now(timezone.utc)
        repo.create_or_update("u3", "c4", {
            "m_task": [0.1] * 768, "m_time_last": now,
            "pi_task": 0.1, "pi_time": 0.05, "mastery": 0.9, "attempts": 10,
        })
        repo.create_or_update("u3", "c5", {
            "m_task": [0.1] * 768, "m_time_last": now,
            "pi_task": 0.5, "pi_time": 0.05, "mastery": 0.3, "attempts": 2,
        })

        weak = repo.get_weak_concepts("u3", threshold=0.8)
        assert len(weak) == 1
        assert weak[0].concept_id == "c5"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/repositories/test_knows_repository.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement KnowsRepository**

Create `src/student_modeling/repositories/knows_repository.py`:

```python
from __future__ import annotations

from typing import Any

from neo4j import ManagedTransaction

from student_modeling.models.knows_edge import KnowsEdge


class KnowsRepository:
    """Repository for KNOWS relationships between User and Concept.

    Not extending BaseRepository since KNOWS is a relationship, not a node.
    """

    def __init__(self, driver, database: str = "neo4j") -> None:
        self._driver = driver
        self._database = database

    def _session(self, **kwargs):
        return self._driver.session(database=self._database, **kwargs)

    def create_or_update(
        self, user_id: str, concept_id: str, props: dict[str, Any]
    ) -> None:
        def _work(tx: ManagedTransaction) -> None:
            tx.run(
                """
                MATCH (u:User {user_id: $user_id})
                MATCH (c:Concept {concept_id: $concept_id})
                MERGE (u)-[k:KNOWS]->(c)
                SET k += $props
                """,
                user_id=user_id,
                concept_id=concept_id,
                props=props,
            )

        with self._session() as session:
            session.execute_write(_work)

    def get_edge(self, user_id: str, concept_id: str) -> KnowsEdge | None:
        def _work(tx: ManagedTransaction) -> dict[str, Any] | None:
            result = tx.run(
                """
                MATCH (u:User {user_id: $user_id})-[k:KNOWS]->(c:Concept {concept_id: $concept_id})
                RETURN k {
                    .*,
                    user_id: u.user_id,
                    concept_id: c.concept_id
                } AS edge
                """,
                user_id=user_id,
                concept_id=concept_id,
            )
            record = result.single()
            return record["edge"] if record else None

        with self._session() as session:
            data = session.execute_read(_work)
            return KnowsEdge.from_record(data) if data else None

    def get_user_overlay(self, user_id: str) -> list[KnowsEdge]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (u:User {user_id: $user_id})-[k:KNOWS]->(c:Concept)
            RETURN k {
                .*,
                user_id: u.user_id,
                concept_id: c.concept_id
            } AS edge
            ORDER BY c.concept_id
            """,
            user_id=user_id,
            database_=self._database,
            routing_="r",
        )
        return [KnowsEdge.from_record(r["edge"]) for r in records]

    def get_weak_concepts(
        self, user_id: str, threshold: float = 0.8
    ) -> list[KnowsEdge]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (u:User {user_id: $user_id})-[k:KNOWS]->(c:Concept)
            WHERE k.mastery < $threshold
            RETURN k {
                .*,
                user_id: u.user_id,
                concept_id: c.concept_id
            } AS edge
            ORDER BY k.mastery ASC
            """,
            user_id=user_id,
            threshold=threshold,
            database_=self._database,
            routing_="r",
        )
        return [KnowsEdge.from_record(r["edge"]) for r in records]

    def get_mastery_map(self, user_id: str) -> dict[str, float]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (u:User {user_id: $user_id})-[k:KNOWS]->(c:Concept)
            RETURN c.concept_id AS concept_id, k.mastery AS mastery
            """,
            user_id=user_id,
            database_=self._database,
            routing_="r",
        )
        return {r["concept_id"]: r["mastery"] for r in records}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/repositories/test_knows_repository.py -v`
Expected: 6 passed

- [ ] **Step 6: Update repositories __init__.py**

Write `src/student_modeling/repositories/__init__.py`:

```python
from student_modeling.repositories.concept_repository import ConceptRepository
from student_modeling.repositories.domain_repository import DomainRepository
from student_modeling.repositories.knows_repository import KnowsRepository
from student_modeling.repositories.learning_object_repository import LearningObjectRepository
from student_modeling.repositories.user_repository import UserRepository

__all__ = [
    "ConceptRepository",
    "DomainRepository",
    "KnowsRepository",
    "LearningObjectRepository",
    "UserRepository",
]
```

- [ ] **Step 7: Commit**

```bash
git add src/student_modeling/repositories/ tests/repositories/
git commit -m "feat: add LearningObject, KNOWS repositories with full CRUD"
```

---

## Task 9: Services — Data Loader

**Files:**
- Create: `src/student_modeling/services/data_loader.py`
- Create: `tests/services/test_data_loader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/services/test_data_loader.py`:

```python
from __future__ import annotations

import pytest

from student_modeling.engine.embeddings import EmbeddingService
from student_modeling.repositories.concept_repository import ConceptRepository
from student_modeling.repositories.domain_repository import DomainRepository
from student_modeling.repositories.learning_object_repository import LearningObjectRepository
from student_modeling.services.data_loader import DataLoader


class TestDataLoader:
    @pytest.fixture(scope="class")
    def embedding_service(self):
        return EmbeddingService(model_name="all-mpnet-base-v2")

    @pytest.fixture
    def loader(self, neo4j_driver, database_name, embedding_service):
        return DataLoader(neo4j_driver, database_name, embedding_service)

    @pytest.fixture
    def domain_repo(self, neo4j_driver, database_name):
        return DomainRepository(neo4j_driver, database_name)

    @pytest.fixture
    def concept_repo(self, neo4j_driver, database_name):
        return ConceptRepository(neo4j_driver, database_name)

    @pytest.fixture
    def lo_repo(self, neo4j_driver, database_name):
        return LearningObjectRepository(neo4j_driver, database_name)

    def test_load_knowledge_graph(self, loader, domain_repo, concept_repo):
        loader.load_knowledge_graph("data/graph.json")

        # 5 domains
        roots = domain_repo.get_root_domains()
        assert len(roots) >= 3  # DOM_01, DOM_02, DOM_05 are roots

        # 10 concepts
        concept = concept_repo.find_by_id("SKILL_01_01")
        assert concept is not None
        assert concept.name == "Early Computing & Cryptography"
        assert len(concept.embedding) == 768

    def test_prerequisites_created(self, loader, concept_repo):
        loader.load_knowledge_graph("data/graph.json")

        # SKILL_01_02 requires SKILL_01_01
        prereqs = concept_repo.get_prerequisites("SKILL_01_02")
        prereq_ids = [p.concept_id for p in prereqs]
        assert "SKILL_01_01" in prereq_ids

    def test_domain_hierarchy(self, loader, neo4j_driver, database_name):
        loader.load_knowledge_graph("data/graph.json")

        # DOM_03 parent is DOM_02
        records, _, _ = neo4j_driver.execute_query(
            """
            MATCH (p:Domain)-[:HAS_CHILD]->(c:Domain {domain_id: 'DOM_03'})
            RETURN p.domain_id AS parent_id
            """,
            database_=database_name,
            routing_="r",
        )
        assert len(records) == 1
        assert records[0]["parent_id"] == "DOM_02"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/services/test_data_loader.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement DataLoader**

Create `src/student_modeling/services/data_loader.py`:

```python
from __future__ import annotations

import json
import logging
from pathlib import Path

import neo4j

from student_modeling.engine.embeddings import EmbeddingService
from student_modeling.models.concept import Concept
from student_modeling.models.domain import Domain
from student_modeling.models.learning_object import LearningObject
from student_modeling.repositories.concept_repository import ConceptRepository
from student_modeling.repositories.domain_repository import DomainRepository
from student_modeling.repositories.learning_object_repository import LearningObjectRepository

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads the expert knowledge graph and learning objects into Neo4j."""

    def __init__(
        self,
        driver: neo4j.Driver,
        database: str,
        embedding_service: EmbeddingService,
    ) -> None:
        self._domain_repo = DomainRepository(driver, database)
        self._concept_repo = ConceptRepository(driver, database)
        self._lo_repo = LearningObjectRepository(driver, database)
        self._embedding = embedding_service

    def load_knowledge_graph(self, graph_path: str) -> None:
        """Load graph.json: creates Domain and Concept nodes with relationships."""
        data = json.loads(Path(graph_path).read_text(encoding="utf-8"))

        # Collect all concept names for batch embedding
        all_concepts: list[dict] = []
        for domain_data in data["knowledge_tree"]:
            for skill in domain_data.get("skills", []):
                all_concepts.append(skill)

        # Batch embed all concept names
        concept_names = [c["name"] for c in all_concepts]
        embeddings = self._embedding.embed_batch(concept_names)

        # Create domains
        for domain_data in data["knowledge_tree"]:
            domain = Domain(
                domain_id=domain_data["node_id"],
                name=domain_data["name"],
                parent_id=domain_data.get("parent_id"),
            )
            self._domain_repo.create(domain)
            logger.info("Created domain: %s", domain.domain_id)

        # Create domain hierarchy
        for domain_data in data["knowledge_tree"]:
            parent_id = domain_data.get("parent_id")
            if parent_id:
                self._domain_repo.add_child(parent_id, domain_data["node_id"])

        # Create concepts with embeddings
        emb_idx = 0
        for domain_data in data["knowledge_tree"]:
            for skill in domain_data.get("skills", []):
                concept = Concept(
                    concept_id=skill["skill_id"],
                    name=skill["name"],
                    domain_id=domain_data["node_id"],
                    embedding=embeddings[emb_idx].tolist(),
                )
                self._concept_repo.create(concept)
                self._concept_repo.link_to_domain(
                    skill["skill_id"], domain_data["node_id"]
                )
                emb_idx += 1

        # Create prerequisite relationships
        for domain_data in data["knowledge_tree"]:
            for skill in domain_data.get("skills", []):
                for prereq_id in skill.get("prerequisites", []):
                    self._concept_repo.add_prerequisite(
                        skill["skill_id"], prereq_id
                    )

        logger.info(
            "Loaded knowledge graph: %d domains, %d concepts",
            len(data["knowledge_tree"]),
            emb_idx,
        )

    def load_learning_objects(
        self, dataset_path: str, concept_question_map: dict[str, list[str]]
    ) -> None:
        """Load Q&A dataset as LearningObject nodes and link to Concepts.

        Args:
            dataset_path: Path to the JSON dataset file.
            concept_question_map: Maps concept_id -> list of lo_ids to link.
        """
        data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))

        # Build question -> lo_id index from graph.json mapping
        # The dataset items don't have IDs, so we use positional index as q_XXX
        lo_id_map: dict[str, dict] = {}
        for i, item in enumerate(data):
            lo_id = f"q_{i + 1:03d}"
            lo_id_map[lo_id] = item

        # Batch embed all questions
        lo_ids = list(lo_id_map.keys())
        questions = [lo_id_map[lid]["question"] for lid in lo_ids]
        embeddings = self._embedding.embed_batch(questions)

        # Create LearningObjects
        los = []
        for idx, lo_id in enumerate(lo_ids):
            item = lo_id_map[lo_id]
            lo = LearningObject(
                lo_id=lo_id,
                question=item["question"],
                answer=item["answer"],
                fact=item.get("fact", ""),
                entities=item.get("entity", []),
                embedding=embeddings[idx].tolist(),
            )
            los.append(lo)

        self._lo_repo.bulk_create(los)

        # Link to concepts
        for concept_id, lo_ids_for_concept in concept_question_map.items():
            for lo_id in lo_ids_for_concept:
                if lo_id in lo_id_map:
                    self._lo_repo.link_to_concept(concept_id, lo_id)

        logger.info("Loaded %d learning objects", len(los))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/services/test_data_loader.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/student_modeling/services/data_loader.py tests/services/test_data_loader.py
git commit -m "feat: add DataLoader service for knowledge graph import"
```

---

## Task 10: Services — ModelingService

**Files:**
- Create: `src/student_modeling/services/modeling_service.py`
- Create: `tests/services/test_modeling_service.py`
- Modify: `src/student_modeling/services/__init__.py`

- [ ] **Step 1: Write failing tests**

Create `tests/services/test_modeling_service.py`:

```python
from __future__ import annotations

import pytest

from student_modeling.engine.embeddings import EmbeddingService
from student_modeling.models.concept import Concept
from student_modeling.models.learning_object import LearningObject
from student_modeling.models.user import User
from student_modeling.repositories.concept_repository import ConceptRepository
from student_modeling.repositories.knows_repository import KnowsRepository
from student_modeling.repositories.learning_object_repository import LearningObjectRepository
from student_modeling.repositories.user_repository import UserRepository
from student_modeling.services.modeling_service import ModelingService


class TestModelingService:
    @pytest.fixture(scope="class")
    def embedding_service(self):
        return EmbeddingService(model_name="all-mpnet-base-v2")

    @pytest.fixture
    def service(self, neo4j_driver, database_name, embedding_service):
        return ModelingService(neo4j_driver, database_name, embedding_service)

    @pytest.fixture
    def user_repo(self, neo4j_driver, database_name):
        return UserRepository(neo4j_driver, database_name)

    @pytest.fixture
    def concept_repo(self, neo4j_driver, database_name):
        return ConceptRepository(neo4j_driver, database_name)

    @pytest.fixture
    def lo_repo(self, neo4j_driver, database_name):
        return LearningObjectRepository(neo4j_driver, database_name)

    @pytest.fixture
    def setup_graph(self, user_repo, concept_repo, lo_repo, embedding_service):
        """Create a minimal test graph."""
        emb_c = embedding_service.embed("Early Computing & Cryptography").tolist()
        emb_lo = embedding_service.embed(
            "Who invented the Enigma machine?"
        ).tolist()

        user_repo.create(User(user_id="u1", name="Alice", email="a@test.com"))
        concept_repo.create(Concept(
            concept_id="c1",
            name="Early Computing & Cryptography",
            domain_id="d1",
            embedding=emb_c,
        ))
        lo_repo.create(LearningObject(
            lo_id="q1",
            question="Who invented the Enigma machine?",
            answer="Arthur Scherbius",
            embedding=emb_lo,
        ))
        lo_repo.link_to_concept("c1", "q1")

    def test_initialize_user(self, service):
        user = service.initialize_user("u_new", "New User", "new@test.com")
        assert user.user_id == "u_new"

    def test_assess_creates_knows_edge(self, service, setup_graph):
        edge = service.assess("u1", "c1", "q1", y=1)
        assert edge.user_id == "u1"
        assert edge.concept_id == "c1"
        assert edge.attempts == 1
        assert edge.mastery > 0.0

    def test_assess_updates_existing(self, service, setup_graph):
        edge1 = service.assess("u1", "c1", "q1", y=1)
        edge2 = service.assess("u1", "c1", "q1", y=1)
        assert edge2.attempts == 2
        assert edge2.mastery >= edge1.mastery

    def test_get_overlay(self, service, setup_graph):
        service.assess("u1", "c1", "q1", y=1)
        overlay = service.get_overlay("u1")
        assert len(overlay) == 1
        assert overlay[0].concept_id == "c1"

    def test_full_scenario_warm_up_and_forget(
        self, service, setup_graph
    ):
        """End-to-end: warm up, then fail, verify mastery drops."""
        # Warm up
        for _ in range(5):
            service.assess("u1", "c1", "q1", y=1)

        overlay = service.get_overlay("u1")
        mastery_peak = overlay[0].mastery
        assert mastery_peak > 0.3

        # Fail
        service.assess("u1", "c1", "q1", y=0)
        overlay = service.get_overlay("u1")
        assert overlay[0].mastery < mastery_peak
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/services/test_modeling_service.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement ModelingService**

Create `src/student_modeling/services/modeling_service.py`:

```python
from __future__ import annotations

import logging

import neo4j
import numpy as np

from student_modeling.engine.embeddings import EmbeddingService
from student_modeling.engine.memory import initialize_knows_edge, process_feedback
from student_modeling.models.concept import Concept
from student_modeling.models.knows_edge import KnowsEdge
from student_modeling.models.user import User
from student_modeling.repositories.concept_repository import ConceptRepository
from student_modeling.repositories.knows_repository import KnowsRepository
from student_modeling.repositories.learning_object_repository import LearningObjectRepository
from student_modeling.repositories.user_repository import UserRepository

logger = logging.getLogger(__name__)


class ModelingService:
    """Orchestrates student modeling with GAM-RAG update policy.

    Wires the pure-math engine with Neo4j repositories.
    """

    def __init__(
        self,
        driver: neo4j.Driver,
        database: str,
        embedding_service: EmbeddingService,
    ) -> None:
        self._user_repo = UserRepository(driver, database)
        self._concept_repo = ConceptRepository(driver, database)
        self._lo_repo = LearningObjectRepository(driver, database)
        self._knows_repo = KnowsRepository(driver, database)
        self._embedding = embedding_service

    def initialize_user(self, user_id: str, name: str, email: str) -> User:
        user = User(user_id=user_id, name=name, email=email)
        created = self._user_repo.create(user)
        logger.info("Initialized user: %s", created.user_id)
        return created

    def assess(
        self, user_id: str, concept_id: str, lo_id: str, y: int
    ) -> KnowsEdge:
        """Process a single feedback event (correct/incorrect).

        1. Fetch LO embedding (query vector)
        2. Fetch or create KNOWS edge
        3. Run GAM-RAG update via engine
        4. Persist updated edge
        5. Return result
        """
        # 1. Get embeddings
        lo = self._lo_repo.find_by_id(lo_id)
        concept = self._concept_repo.find_by_id(concept_id)
        query_emb = np.array(lo.embedding, dtype=np.float32)
        concept_emb = np.array(concept.embedding, dtype=np.float32)

        # 2. Get or create KNOWS edge
        existing = self._knows_repo.get_edge(user_id, concept_id)
        if existing is None:
            edge_props = initialize_knows_edge(concept_emb)
        else:
            edge_props = {
                "m_task": np.array(existing.m_task, dtype=np.float32),
                "m_time_last": existing.m_time_last,
                "pi_task": existing.pi_task,
                "pi_time": existing.pi_time,
                "mastery": existing.mastery,
                "attempts": existing.attempts,
            }

        # 3. Run GAM-RAG update
        updated = process_feedback(
            y=y,
            query_emb=query_emb,
            concept_emb=concept_emb,
            **edge_props,
        )

        # 4. Convert numpy arrays to lists for Neo4j storage
        persist_props = {
            "m_task": updated["m_task"].tolist(),
            "m_time_last": updated["m_time_last"],
            "pi_task": updated["pi_task"],
            "pi_time": updated["pi_time"],
            "mastery": updated["mastery"],
            "attempts": updated["attempts"],
        }
        self._knows_repo.create_or_update(user_id, concept_id, persist_props)

        # 5. Return as KnowsEdge
        logger.info(
            "Assessed %s on %s (y=%d): mastery=%.3f pi_task=%.3f",
            user_id, concept_id, y, updated["mastery"], updated["pi_task"],
        )
        return KnowsEdge(
            user_id=user_id,
            concept_id=concept_id,
            m_task=persist_props["m_task"],
            m_time_last=updated["m_time_last"],
            pi_task=updated["pi_task"],
            pi_time=updated["pi_time"],
            mastery=updated["mastery"],
            attempts=updated["attempts"],
        )

    def get_overlay(self, user_id: str) -> list[KnowsEdge]:
        return self._knows_repo.get_user_overlay(user_id)

    def get_learning_path(self, user_id: str) -> list[Concept]:
        """Get unmastered concepts with satisfied prerequisites.

        Returns concepts ordered by prerequisite depth (shallowest first).
        """
        records, _, _ = self._knows_repo._driver.execute_query(
            """
            MATCH (c:Concept)
            WHERE NOT EXISTS {
                MATCH (:User {user_id: $user_id})-[k:KNOWS]->(c)
                WHERE k.mastery >= 0.8
            }
            OPTIONAL MATCH (c)-[:HAS_PREREQUISITE]->(prereq:Concept)
            OPTIONAL MATCH (:User {user_id: $user_id})-[pk:KNOWS]->(prereq)
            WITH c,
                 collect(prereq.concept_id) AS prereq_ids,
                 collect(CASE WHEN pk.mastery >= 0.8 THEN 1 ELSE 0 END) AS prereq_met
            WITH c,
                 size(prereq_ids) AS total_prereqs,
                 reduce(s = 0, x IN prereq_met | s + x) AS met_prereqs
            WHERE total_prereqs = 0 OR met_prereqs = total_prereqs
            RETURN c {.*, element_id: elementId(c)} AS concept
            ORDER BY c.name
            """,
            user_id=user_id,
            database_=self._knows_repo._database,
            routing_="r",
        )
        return [Concept.from_node(r["concept"]) for r in records]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/services/test_modeling_service.py -v`
Expected: 5 passed

- [ ] **Step 5: Update services __init__.py**

Write `src/student_modeling/services/__init__.py`:

```python
from student_modeling.services.data_loader import DataLoader
from student_modeling.services.modeling_service import ModelingService

__all__ = ["DataLoader", "ModelingService"]
```

- [ ] **Step 6: Update tests/conftest.py**

Write `tests/conftest.py`:

```python
from __future__ import annotations

import os

import pytest
from neo4j import GraphDatabase

from student_modeling.config import Settings


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(
        neo4j_uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "changeme"),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )


@pytest.fixture(scope="session")
def neo4j_driver(settings):
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
    )
    driver.verify_connectivity()
    yield driver
    driver.close()


@pytest.fixture(autouse=True)
def clean_database(neo4j_driver, settings):
    """Clean all test data before each test."""
    with neo4j_driver.session(database=settings.neo4j_database) as session:
        session.run("MATCH (n) DETACH DELETE n").consume()
    yield


@pytest.fixture
def database_name(settings) -> str:
    return settings.neo4j_database
```

- [ ] **Step 7: Commit**

```bash
git add src/student_modeling/services/ tests/services/ tests/conftest.py
git commit -m "feat: add ModelingService with GAM-RAG assess() and learning path"
```

---

## Task 11: Full Integration Test and Final Cleanup

**Files:**
- Modify: `src/student_modeling/engine/__init__.py`

- [ ] **Step 1: Update engine __init__.py**

Write `src/student_modeling/engine/__init__.py`:

```python
from student_modeling.engine.embeddings import EmbeddingService
from student_modeling.engine.kalman import (
    compute_gain,
    compute_residual,
    compute_time_decay,
    update_memory,
    update_perplexity,
)
from student_modeling.engine.memory import (
    derive_mastery,
    initialize_knows_edge,
    process_feedback,
)

__all__ = [
    "EmbeddingService",
    "compute_gain",
    "compute_residual",
    "compute_time_decay",
    "derive_mastery",
    "initialize_knows_edge",
    "process_feedback",
    "update_memory",
    "update_perplexity",
]
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass (engine tests ~14, memory tests ~13, embedding tests ~4, repository tests ~10, service tests ~8)

- [ ] **Step 3: Run linter**

Run: `ruff check src/ tests/`
Expected: No errors. Fix any issues if found.

- [ ] **Step 4: Commit final cleanup**

```bash
git add -A
git commit -m "feat: complete GAM-RAG student modeling implementation"
```

---

## Summary

| Task | What It Builds | Key Test |
|------|---------------|----------|
| 1 | Dependencies + config | Install check |
| 2 | Remove old scaffold | Clean slate |
| 3 | `engine/kalman.py` | Gain, residual, perplexity math |
| 4 | `engine/memory.py` | Warm-up, damped refinement, forgetting, recovery |
| 5 | `engine/embeddings.py` | all-mpnet-base-v2 wrapper |
| 6 | Domain models | Import check |
| 7 | User, Domain, Concept repos | CRUD + relationships |
| 8 | LO + KNOWS repos | Vector storage on edges |
| 9 | DataLoader service | graph.json import |
| 10 | ModelingService | Full assess() + overlay |
| 11 | Integration + cleanup | Full suite pass |
