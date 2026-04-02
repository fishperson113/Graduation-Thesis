from __future__ import annotations

import pytest

from student_modeling.engine.embeddings import EmbeddingService
from student_modeling.models.concept import Concept
from student_modeling.models.learning_object import LearningObject
from student_modeling.models.user import User
from student_modeling.repositories.concept_repository import ConceptRepository
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
