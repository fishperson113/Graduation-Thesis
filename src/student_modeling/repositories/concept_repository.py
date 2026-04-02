from __future__ import annotations

from typing import Any

from neo4j import ManagedTransaction

from student_modeling.models.concept import Concept
from student_modeling.repositories.base import BaseRepository


class ConceptRepository(BaseRepository):
    def create(self, concept: Concept) -> Concept:
        def _work(tx: ManagedTransaction) -> dict[str, Any]:
            result = tx.run(
                """
                CREATE (c:Concept {
                    concept_id: $concept_id,
                    name: $name,
                    domain_id: $domain_id,
                    description: $description,
                    embedding: $embedding
                })
                RETURN c {.*, element_id: elementId(c)} AS concept
                """,
                **concept.to_properties(),
            )
            return result.single(strict=True)["concept"]

        with self._session() as session:
            data = session.execute_write(_work)
            return Concept.from_node(data)

    def find_by_id(self, concept_id: str) -> Concept | None:
        def _work(tx: ManagedTransaction) -> dict[str, Any] | None:
            result = tx.run(
                """
                MATCH (c:Concept {concept_id: $concept_id})
                RETURN c {.*, element_id: elementId(c)} AS concept
                """,
                concept_id=concept_id,
            )
            record = result.single()
            return record["concept"] if record else None

        with self._session() as session:
            data = session.execute_read(_work)
            return Concept.from_node(data) if data else None

    def delete(self, concept_id: str) -> bool:
        def _work(tx: ManagedTransaction) -> int:
            result = tx.run(
                """
                MATCH (c:Concept {concept_id: $concept_id})
                DETACH DELETE c
                RETURN count(*) AS deleted
                """,
                concept_id=concept_id,
            )
            return result.single(strict=True)["deleted"]

        with self._session() as session:
            return session.execute_write(_work) > 0

    def add_prerequisite(self, concept_id: str, prerequisite_id: str) -> None:
        def _work(tx: ManagedTransaction) -> None:
            tx.run(
                """
                MATCH (c:Concept {concept_id: $concept_id})
                MATCH (p:Concept {concept_id: $prerequisite_id})
                MERGE (c)-[:HAS_PREREQUISITE]->(p)
                """,
                concept_id=concept_id,
                prerequisite_id=prerequisite_id,
            )

        with self._session() as session:
            session.execute_write(_work)

    def get_prerequisites(self, concept_id: str) -> list[Concept]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (c:Concept {concept_id: $concept_id})-[:HAS_PREREQUISITE]->(p:Concept)
            RETURN p {.*, element_id: elementId(p)} AS concept
            ORDER BY p.name
            """,
            concept_id=concept_id,
            database_=self._database,
            routing_="r",
        )
        return [Concept.from_node(r["concept"]) for r in records]

    def get_by_domain(self, domain_id: str) -> list[Concept]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (c:Concept {domain_id: $domain_id})
            RETURN c {.*, element_id: elementId(c)} AS concept
            ORDER BY c.name
            """,
            domain_id=domain_id,
            database_=self._database,
            routing_="r",
        )
        return [Concept.from_node(r["concept"]) for r in records]

    def link_to_domain(self, concept_id: str, domain_id: str) -> None:
        def _work(tx: ManagedTransaction) -> None:
            tx.run(
                """
                MATCH (d:Domain {domain_id: $domain_id})
                MATCH (c:Concept {concept_id: $concept_id})
                MERGE (d)-[:CONTAINS]->(c)
                """,
                domain_id=domain_id,
                concept_id=concept_id,
            )

        with self._session() as session:
            session.execute_write(_work)
