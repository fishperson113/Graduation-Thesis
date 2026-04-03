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


class TestProcessFeedback:
    """These tests prove the GAM-RAG theory works for student modeling."""

    def _make_concept_emb(self, seed=42):
        rng = np.random.default_rng(seed)
        emb = rng.standard_normal(768).astype(np.float32)
        return emb / np.linalg.norm(emb)

    def test_single_correct_answer_increases_mastery(self):
        concept_emb = self._make_concept_emb()
        edge = initialize_knows_edge(concept_emb)
        query_emb = concept_emb.copy()

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

        for _ in range(20):
            edge = process_feedback(
                y=1, query_emb=query_emb, concept_emb=concept_emb, **edge
            )

        mastery_before = edge["mastery"]

        edge = process_feedback(
            y=1, query_emb=query_emb, concept_emb=concept_emb, **edge
        )
        mastery_after = edge["mastery"]

        assert abs(mastery_after - mastery_before) < 0.05

    def test_forgetting_repulsion(self):
        """Master a concept, then fail -> mastery should drop (repulsion)."""
        concept_emb = self._make_concept_emb()
        edge = initialize_knows_edge(concept_emb)
        query_emb = concept_emb.copy()

        for _ in range(15):
            edge = process_feedback(
                y=1, query_emb=query_emb, concept_emb=concept_emb, **edge
            )
        mastery_peak = edge["mastery"]
        assert mastery_peak > 0.5

        edge = process_feedback(
            y=0, query_emb=query_emb, concept_emb=concept_emb, **edge
        )
        assert edge["mastery"] < mastery_peak

    def test_relearning_recovery(self):
        """After forgetting, correct answers should recover mastery."""
        concept_emb = self._make_concept_emb()
        edge = initialize_knows_edge(concept_emb)
        query_emb = concept_emb.copy()

        for _ in range(15):
            edge = process_feedback(
                y=1, query_emb=query_emb, concept_emb=concept_emb, **edge
            )

        for _ in range(3):
            edge = process_feedback(
                y=0, query_emb=query_emb, concept_emb=concept_emb, **edge
            )
        mastery_after_forget = edge["mastery"]

        for _ in range(5):
            edge = process_feedback(
                y=1, query_emb=query_emb, concept_emb=concept_emb, **edge
            )

        assert edge["mastery"] > mastery_after_forget

    def test_time_decay_increases_perplexity(self):
        """Long time gap increases perplexity and reduces mastery."""
        from datetime import timedelta

        concept_emb = self._make_concept_emb()
        edge = initialize_knows_edge(concept_emb)
        query_emb = concept_emb.copy()

        for _ in range(10):
            edge = process_feedback(
                y=1, query_emb=query_emb, concept_emb=concept_emb, **edge
            )

        mastery_before_gap = edge["mastery"]

        # Simulate 30 days passing
        edge["m_time_last"] = datetime.now(timezone.utc) - timedelta(days=30)

        edge_after = process_feedback(
            y=1, query_emb=query_emb, concept_emb=concept_emb, **edge
        )
        assert edge_after["attempts"] == edge["attempts"] + 1
        # Even a correct answer after a long gap should yield lower mastery
        # than the peak before the gap, because time decay raised perplexity.
        assert edge_after["mastery"] < mastery_before_gap

    def test_different_embeddings_failure_drops_mastery(self):
        """When query_emb != concept_emb, failure must still reduce mastery.

        This is the scenario that caused a real bug: memory drifts toward
        query during warm-up, then a failure pushes it back — which must
        not accidentally increase alignment with concept_emb.
        """
        rng = np.random.default_rng(99)
        concept_emb = self._make_concept_emb(seed=42)
        # Create a related but distinct query embedding (cosine ~0.3-0.5)
        query_emb = concept_emb + 0.8 * rng.standard_normal(768).astype(np.float32)
        query_emb = (query_emb / np.linalg.norm(query_emb)).astype(np.float32)

        edge = initialize_knows_edge(concept_emb)

        for _ in range(10):
            edge = process_feedback(
                y=1, query_emb=query_emb, concept_emb=concept_emb, **edge
            )
        mastery_peak = edge["mastery"]
        assert mastery_peak > 0.3

        # Single failure must drop mastery
        edge = process_feedback(
            y=0, query_emb=query_emb, concept_emb=concept_emb, **edge
        )
        assert edge["mastery"] < mastery_peak

    def test_different_embeddings_success_increases_mastery(self):
        """When query_emb != concept_emb, success must still increase mastery."""
        rng = np.random.default_rng(99)
        concept_emb = self._make_concept_emb(seed=42)
        query_emb = concept_emb + 0.8 * rng.standard_normal(768).astype(np.float32)
        query_emb = (query_emb / np.linalg.norm(query_emb)).astype(np.float32)

        edge = initialize_knows_edge(concept_emb)

        edge_after = process_feedback(
            y=1, query_emb=query_emb, concept_emb=concept_emb, **edge
        )
        assert edge_after["mastery"] > edge["mastery"]

    def test_multiple_los_per_concept(self):
        """Different LOs for the same concept should all drive mastery up."""
        rng = np.random.default_rng(77)
        concept_emb = self._make_concept_emb(seed=42)

        # Three distinct query embeddings (different questions about same concept)
        queries = []
        for i in range(3):
            q = concept_emb + 0.6 * rng.standard_normal(768).astype(np.float32)
            q = (q / np.linalg.norm(q)).astype(np.float32)
            queries.append(q)

        edge = initialize_knows_edge(concept_emb)

        # Answer each LO correctly in rotation
        for _ in range(4):
            for q in queries:
                prev_mastery = edge["mastery"]
                edge = process_feedback(
                    y=1, query_emb=q, concept_emb=concept_emb, **edge
                )
                assert edge["mastery"] >= prev_mastery - 0.01  # allow tiny float noise

        assert edge["mastery"] > 0.5

    def test_repeated_failures_drop_mastery_monotonically(self):
        """Sustained failures should reduce mastery monotonically.

        Mastery is driven by pi_task (which converges to a fixed point
        under repeated failures due to process noise Q). The floor is
        ~0.71 with default params — this models partial retention.
        """
        concept_emb = self._make_concept_emb()
        edge = initialize_knows_edge(concept_emb)
        query_emb = concept_emb.copy()

        # Warm up first
        for _ in range(10):
            edge = process_feedback(
                y=1, query_emb=query_emb, concept_emb=concept_emb, **edge
            )
        mastery_peak = edge["mastery"]
        assert mastery_peak > 0.5

        # Sustained failures: mastery should drop monotonically
        prev = mastery_peak
        for _ in range(20):
            edge = process_feedback(
                y=0, query_emb=query_emb, concept_emb=concept_emb, **edge
            )
            assert edge["mastery"] <= prev + 1e-9  # monotonically decreasing
            prev = edge["mastery"]

        assert edge["mastery"] < mastery_peak

    def test_alternating_feedback_stays_bounded(self):
        """Alternating correct/incorrect should keep mastery in a bounded range.

        With default params, alternating feedback oscillates between ~0.72-0.76
        as pi_task oscillates around its alternating fixed point.
        """
        concept_emb = self._make_concept_emb()
        edge = initialize_knows_edge(concept_emb)
        query_emb = concept_emb.copy()

        masteries = []
        for i in range(30):
            edge = process_feedback(
                y=i % 2, query_emb=query_emb, concept_emb=concept_emb, **edge
            )
            masteries.append(edge["mastery"])

        # Should settle into a bounded oscillation, not diverge
        final_range = masteries[-10:]
        assert max(final_range) - min(final_range) < 0.05  # tight oscillation
        assert 0.3 < min(final_range)  # not collapsed to zero
        assert max(final_range) < 0.95  # not pinned at ceiling
