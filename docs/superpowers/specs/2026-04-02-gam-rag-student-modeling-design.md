# Graph-Based Student Modeling with GAM-RAG Update Policy

**Date:** 2026-04-02
**Status:** Design Approved
**Scope:** Prove the theory — validate GAM-RAG Kalman-gain dynamics as a student modeling update policy

---

## 1. Overview

Build a graph-based student modeling system where a student's knowledge is represented as a dynamic overlay on top of an expert knowledge graph. The overlay evolves using an adapted GAM-RAG update policy (Kalman-inspired gain rule) that provides:

- **Fast Warm-up:** Rapid mastery increase for new concepts (high perplexity → high gain)
- **Damped Refinement:** Stable mastery for well-known concepts (low perplexity → low gain)
- **Forgetting via Repulsion:** Mastery decay when a student fails previously-mastered concepts
- **Re-learning Recovery:** Fast recovery driven by increased perplexity after decay

**Reference paper:** [GAM-RAG](https://arxiv.org/abs/2603.01783)

**Future direction:** This module becomes a FastMCP tool for a main AI Agent, evolving into a full GraphRAG + MLOps pipeline. For now, focus is on proving the mathematical theory works for student modeling.

---

## 2. Architecture: Layered Domain Engine

```
repositories/    → Neo4j CRUD
engine/          → Pure math (no DB dependency) — future MCP core
services/        → Orchestration (wires engine + repos)
```

The `engine/` layer is pure Python with numpy — takes vectors/scalars in, returns vectors/scalars out. Zero knowledge of Neo4j. This is the module that becomes the core of the MCP tool later.

---

## 3. Graph Schema

### 3.1 Expert Model (Static Knowledge Graph)

```
Node Labels:
  (:Domain {domain_id, name, parent_id})
  (:Concept {concept_id, name, description, domain_id, embedding: [float x 768]})
  (:LearningObject {lo_id, question, answer, fact, entities: [str], embedding: [float x 768]})

Relationships:
  (:Domain)-[:HAS_CHILD]->(:Domain)
  (:Domain)-[:CONTAINS]->(:Concept)
  (:Concept)-[:HAS_PREREQUISITE]->(:Concept)
  (:Concept)-[:COVERS]->(:LearningObject)
```

### 3.2 Student Overlay (Dynamic)

```
Node Labels:
  (:User {user_id, name, email, created_at})

Relationships:
  (:User)-[:KNOWS {
      m_task: [float x 768],    — task-oriented memory vector
      m_time_last: datetime,     — last interaction timestamp
      pi_task: float,            — task perplexity in [0,1]
      pi_time: float,            — time perplexity in [0,1]
      mastery: float,            — derived mastery score in [0,1]
      attempts: int              — total interaction count
  }]->(:Concept)
```

### 3.3 Embedding Model

`all-mpnet-base-v2` from sentence-transformers (768-dim). Same as the GAM-RAG paper. Chosen for future GraphRAG compatibility.

### 3.4 Data Sources

- `data/graph.json` — Knowledge graph structure (5 domains, 10 concepts, prerequisite chains, question mappings)
- `data/it_software_dev (2).json` — Q&A dataset (questions with answers, facts, entities)

---

## 4. GAM-RAG Update Policy (Adapted for Student Modeling)

### 4.1 Feedback Signal

Binary: `y in {0, 1}` — correct/incorrect on a LearningObject (question).

No LLM judge in this phase. The binary outcome IS the feedback signal.

### 4.2 Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| `R_POS` | 0.5 | Observation noise for correct answer (y=1) |
| `R_NEG` | 1.0 | Observation noise for incorrect answer (y=0) |
| `Q_TASK` | 0.05 | Process noise for task perplexity |
| `Q_TIME` | 0.05 | Process noise for time perplexity |
| `PI_INIT` | 1.0 | Initial perplexity (maximum uncertainty) |
| `TIME_DECAY_RATE` | configurable | Rate at which pi_time increases per day of absence |

### 4.3 Kalman Gain

$$K = \frac{\pi}{\pi + R_i}$$

Where $R_i = R_{POS}$ if $y = 1$, else $R_i = R_{NEG}$.

- When $\pi \approx 1$ (new concept): $K$ is large → fast updates (warm-up)
- When $\pi \to Q$ (stable): $K$ is small → damped refinement

### 4.4 Residual

$$e = y - \cos(q, m_{old})$$

Where $q$ is the LearningObject embedding (query vector) and $m_{old}$ is the current task memory vector.

- $e > 0$: pull memory toward query direction (attraction)
- $e < 0$: push memory away from query direction (repulsion)

### 4.5 Memory Update

$$m_{new} = \text{normalize}(m_{old} + K \cdot e \cdot q)$$

L2 normalization post-update ensures $\|m\|_2 = 1$.

### 4.6 Perplexity Update

$$\pi_{new} = \text{clip}_{[0,1]}((1 - K) \cdot \pi_{old} + Q)$$

Process noise $Q > 0$ prevents perplexity from collapsing to zero, maintaining a permanent "window of adaptability."

### 4.7 Time Decay (Forgetting Curve)

Adapted from the paper's temporal memory to fit student modeling:

```
pi_time_decayed = clip[0,1](pi_time + time_decay_rate * days_elapsed)
pi_task_decayed = clip[0,1](pi_task + time_decay_rate * days_elapsed)
```

Both perplexities increase with time absence — modeling the forgetting curve. This means:
- Long absence → higher perplexity → higher gain on next interaction
- Even without explicit failure, temporal absence reopens the "window of adaptability"

### 4.8 Mastery Derivation

$$\text{mastery} = \text{clip}_{[0,1]}((1 - \pi_{task}) \cdot \cos(m_{task}, c_{emb}) \cdot (1 - \pi_{time}))$$

Where $c_{emb}$ is the concept's original embedding. Mastery is high only when:
1. Task perplexity is low (confident from interactions)
2. Memory vector aligns with concept embedding (correct knowledge direction)
3. Time perplexity is low (recent interaction)

### 4.9 KNOWS Edge Initialization

On first interaction with a concept:
- `m_task` = concept embedding (normalized)
- `m_time_last` = now
- `pi_task` = 1.0 (maximum uncertainty)
- `pi_time` = 1.0 (maximum uncertainty)
- `mastery` = 0.0
- `attempts` = 0

### 4.10 Full Update Cycle (in `process_feedback`)

1. Apply time decay to `pi_time` and `pi_task` based on elapsed time since `m_time_last`
2. Select observation noise: $R = R_{POS}$ if $y = 1$, else $R = R_{NEG}$
3. Compute Kalman gain: $K = \pi_{task} / (\pi_{task} + R)$
4. Compute residual: $e = y - \cos(q, m_{task})$
5. Update memory vector: $m_{task} = \text{normalize}(m_{task} + K \cdot e \cdot q)$
6. Update task perplexity: $\pi_{task} = \text{clip}((1 - K) \cdot \pi_{task} + Q_{task})$
7. Update time: `m_time_last = now`
8. Reset `pi_time = Q_TIME` (fresh interaction restores time confidence to floor)
9. Derive new mastery score
10. Increment attempts

---

## 5. Dynamic Competence and Knowledge Decay

Three mechanisms ensure the student model treats competence as fluid:

### 5.1 Temporal Awareness

`pi_time` increases passively as time passes since last interaction. This tracks not just *what* the student knows, but *how recently* they demonstrated mastery. Time decay also feeds into `pi_task` — because temporal absence itself is evidence of potential forgetting.

### 5.2 Process Noise Prevents Lock-in

$Q > 0$ ensures that even for high-mastery concepts, perplexity never collapses to zero. The perplexity floor is $Q$ (from paper Corollary 1: $\pi \in [Q, 1]$). This keeps a permanent "window of adaptability" open for every concept.

### 5.3 Repulsion on Forgetting

When a student who previously mastered a concept fails:
- $\cos(q, m_{old})$ is high (strong prior alignment)
- $e = 0 - \cos(q, m_{old})$ is strongly negative
- Memory vector is pushed away from the query direction
- Mastery drops accordingly

### 5.4 Re-learning Recovery

After repulsion:
- `pi_task` has increased (via perplexity update with strong negative residual + process noise)
- Higher perplexity → higher Kalman gain on next positive signal
- Correct answers drive rapid mastery recovery (fast warm-up dynamics)

---

## 6. File Structure

### 6.1 New Files

```
src/student_modeling/
  ├── engine/
  │   ├── __init__.py
  │   ├── kalman.py              — gain, residual, perplexity, time decay
  │   ├── memory.py              — initialize, derive_mastery, process_feedback
  │   └── embeddings.py          — all-mpnet-base-v2 wrapper
  ├── models/
  │   ├── domain.py              — Domain dataclass
  │   ├── concept.py             — Concept dataclass
  │   ├── learning_object.py     — LearningObject dataclass
  │   ├── user.py                — User dataclass
  │   └── knows_edge.py          — KnowsEdge dataclass (GAM-RAG properties)
  ├── repositories/
  │   ├── domain_repository.py
  │   ├── concept_repository.py
  │   ├── learning_object_repository.py
  │   ├── user_repository.py
  │   └── knows_repository.py
  └── services/
      ├── modeling_service.py    — assess(), get_overlay(), get_learning_path()
      └── data_loader.py         — load graph.json + dataset into Neo4j

tests/
  ├── engine/
  │   ├── test_kalman.py         — pure math validation
  │   ├── test_memory.py         — full cycle theory tests
  │   └── test_embeddings.py
  ├── repositories/
  │   ├── test_concept_repository.py
  │   ├── test_knows_repository.py
  │   └── test_user_repository.py
  └── services/
      ├── test_modeling_service.py
      └── test_data_loader.py
```

### 6.2 Files to Remove

- `models/student.py`, `models/course.py`, `models/skill.py`, `models/knowledge_state.py`
- `repositories/student_repository.py`, `repositories/course_repository.py`, `repositories/skill_repository.py`, `repositories/knowledge_state_repository.py`
- `services/student_service.py`
- `tests/repositories/test_student_repository.py`

### 6.3 Files to Modify

- `config.py` — add embedding model name, time decay rate, GAM-RAG constants
- `pyproject.toml` — add `sentence-transformers`, `numpy` dependencies
- `tests/conftest.py` — update fixtures for new models

### 6.4 Files to Keep As-Is

- `database.py` — sync + async Neo4j connection management
- `exceptions.py` — custom exception hierarchy
- `repositories/base.py` — BaseRepository abstract class
- `docker-compose.yml` — Neo4j 5 setup

---

## 7. Dependencies

Add to `pyproject.toml`:
```toml
dependencies = [
    "neo4j>=5.0",
    "pydantic-settings>=2.0",
    "sentence-transformers>=2.0",
    "numpy>=1.24",
]
```

---

## 8. Testing Strategy

### 8.1 Engine Tests (Prove the Theory — No DB Required)

**`test_kalman.py`** — Pure math:
- Gain high/low perplexity (warm-up vs damped)
- Asymmetric noise (K_pos > K_neg for same pi)
- Residual sign (positive/negative)
- Memory update direction (attraction/repulsion)
- Memory normalization (||m|| = 1)
- Perplexity decrease and floor
- Time decay

**`test_memory.py`** — Full cycles:
- Initialize edge defaults
- Warm-up sequence (5 correct → mastery rises rapidly)
- Damped refinement (20+ correct → changes slow)
- Forgetting repulsion (master + fail → mastery drops)
- Re-learning recovery (forget + correct → fast recovery)
- Time decay mastery (long absence → mastery decreases)
- Full process_feedback cycle

### 8.2 Integration Tests (DB Required)

**`test_modeling_service.py`:**
- assess creates KNOWS edge on first interaction
- assess updates existing edge
- overlay reflects correct mastery state
- learning path respects prerequisites
- full scenario: load graph → user → assess multiple concepts → verify overlay

**`test_data_loader.py`:**
- load graph.json creates correct nodes and relationships
- load dataset creates LearningObjects with embeddings

---

## 9. LLM Provider

Mistral (via `MISTRAL_API_KEY`) reserved for future use:
- LLM judge for richer feedback signals
- Fine-tuning on domain-specific assessment
- GraphRAG query processing

Not needed for "prove the theory" phase — binary feedback is sufficient.

---

## 10. Future Direction (Out of Scope for This Phase)

- FastMCP tool wrapping `ModelingService.assess()` and `get_overlay()`
- GraphRAG retrieval using memory states as query-aligned priors
- MLOps pipeline (model versioning, A/B testing, monitoring)
- Vector index on Neo4j for embedding-based concept search
- Mistral LLM judge replacing binary feedback
