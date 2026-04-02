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
