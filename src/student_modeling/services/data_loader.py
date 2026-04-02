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
