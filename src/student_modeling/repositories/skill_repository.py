from __future__ import annotations

from typing import Any

from neo4j import ManagedTransaction

from student_modeling.exceptions import EntityNotFoundError
from student_modeling.models.skill import Skill
from student_modeling.repositories.base import BaseRepository


class SkillRepository(BaseRepository):
    def create(self, skill: Skill) -> Skill:
        def _work(tx: ManagedTransaction) -> dict[str, Any]:
            result = tx.run(
                """
                CREATE (s:Skill {
                    skill_id: $skill_id,
                    name: $name,
                    category: $category,
                    description: $description
                })
                RETURN s {.*, element_id: elementId(s)} AS skill
                """,
                **skill.to_properties(),
            )
            return result.single(strict=True)["skill"]

        with self._session() as session:
            data = session.execute_write(_work)
            return Skill.from_node(data)

    def find_by_id(self, skill_id: str) -> Skill | None:
        def _work(tx: ManagedTransaction) -> dict[str, Any] | None:
            result = tx.run(
                """
                MATCH (s:Skill {skill_id: $skill_id})
                RETURN s {.*, element_id: elementId(s)} AS skill
                """,
                skill_id=skill_id,
            )
            record = result.single()
            return record["skill"] if record else None

        with self._session() as session:
            data = session.execute_read(_work)
            return Skill.from_node(data) if data else None

    def delete(self, skill_id: str) -> bool:
        def _work(tx: ManagedTransaction) -> int:
            result = tx.run(
                """
                MATCH (s:Skill {skill_id: $skill_id})
                WITH s, s.skill_id AS sid
                DETACH DELETE s
                RETURN count(sid) AS deleted
                """,
                skill_id=skill_id,
            )
            return result.single(strict=True)["deleted"]

        with self._session() as session:
            return session.execute_write(_work) > 0

    def add_prerequisite(self, skill_id: str, prerequisite_id: str) -> None:
        def _work(tx: ManagedTransaction) -> None:
            result = tx.run(
                """
                MATCH (s:Skill {skill_id: $skill_id})
                MATCH (p:Skill {skill_id: $prerequisite_id})
                MERGE (p)-[:PREREQUISITE_OF]->(s)
                RETURN s.skill_id AS sid
                """,
                skill_id=skill_id,
                prerequisite_id=prerequisite_id,
            )
            if result.single() is None:
                raise EntityNotFoundError("Skill", f"{skill_id} or {prerequisite_id}")

        with self._session() as session:
            session.execute_write(_work)

    def get_prerequisites(self, skill_id: str) -> list[Skill]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (p:Skill)-[:PREREQUISITE_OF]->(s:Skill {skill_id: $skill_id})
            RETURN p {.*, element_id: elementId(p)} AS skill
            ORDER BY p.name
            """,
            skill_id=skill_id,
            database_=self._database,
            routing_="r",
        )
        return [Skill.from_node(r["skill"]) for r in records]
