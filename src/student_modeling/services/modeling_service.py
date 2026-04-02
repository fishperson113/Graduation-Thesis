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
