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
        from student_modeling.config import get_settings

        settings = get_settings()
        api_key = settings.huggingface_api_key.get_secret_value()
        return EmbeddingService(api_key=api_key)

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
