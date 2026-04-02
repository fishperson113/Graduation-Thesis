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
