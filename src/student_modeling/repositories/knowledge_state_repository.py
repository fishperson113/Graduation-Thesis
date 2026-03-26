from __future__ import annotations

from typing import Any

from neo4j import ManagedTransaction

from student_modeling.models.knowledge_state import KnowledgeState
from student_modeling.repositories.base import BaseRepository


class KnowledgeStateRepository(BaseRepository):
    def create(self, entity: Any) -> Any:
        raise NotImplementedError("Use update_mastery to create/update knowledge states")

    def find_by_id(self, entity_id: str) -> Any | None:
        raise NotImplementedError("Use get_student_knowledge instead")

    def delete(self, entity_id: str) -> bool:
        raise NotImplementedError("Knowledge states are deleted with their parent nodes")

    def update_mastery(
        self,
        student_id: str,
        skill_id: str,
        mastery_level: float,
        confidence: float,
    ) -> KnowledgeState:
        def _work(tx: ManagedTransaction) -> dict[str, Any]:
            result = tx.run(
                """
                MATCH (s:Student {student_id: $student_id})
                MATCH (sk:Skill {skill_id: $skill_id})
                MERGE (s)-[r:HAS_KNOWLEDGE]->(sk)
                SET r.mastery_level = $mastery_level,
                    r.confidence = $confidence,
                    r.last_assessed_at = datetime(),
                    r.attempts = coalesce(r.attempts, 0) + 1
                RETURN s.student_id AS student_id,
                       sk.skill_id AS skill_id,
                       r.mastery_level AS mastery_level,
                       r.confidence AS confidence,
                       r.last_assessed_at AS last_assessed_at,
                       r.attempts AS attempts
                """,
                student_id=student_id,
                skill_id=skill_id,
                mastery_level=mastery_level,
                confidence=confidence,
            )
            return dict(result.single(strict=True))

        with self._session() as session:
            data = session.execute_write(_work)
            return KnowledgeState.from_record(data)

    def get_student_knowledge(self, student_id: str) -> list[KnowledgeState]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (s:Student {student_id: $student_id})-[r:HAS_KNOWLEDGE]->(sk:Skill)
            RETURN s.student_id AS student_id,
                   sk.skill_id AS skill_id,
                   r.mastery_level AS mastery_level,
                   r.confidence AS confidence,
                   r.last_assessed_at AS last_assessed_at,
                   r.attempts AS attempts
            ORDER BY r.mastery_level DESC
            """,
            student_id=student_id,
            database_=self._database,
            routing_="r",
        )
        return [KnowledgeState.from_record(dict(r)) for r in records]

    def get_skill_mastery_distribution(self, skill_id: str) -> dict[str, Any]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (s:Student)-[r:HAS_KNOWLEDGE]->(sk:Skill {skill_id: $skill_id})
            RETURN count(s) AS total_students,
                   avg(r.mastery_level) AS avg_mastery,
                   min(r.mastery_level) AS min_mastery,
                   max(r.mastery_level) AS max_mastery,
                   avg(r.confidence) AS avg_confidence
            """,
            skill_id=skill_id,
            database_=self._database,
            routing_="r",
        )
        record = records[0] if records else None
        if record is None:
            return {"total_students": 0}
        return dict(record)
