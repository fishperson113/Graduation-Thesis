from __future__ import annotations

from typing import Any

from neo4j import ManagedTransaction

from student_modeling.models.domain import Domain
from student_modeling.repositories.base import BaseRepository


class DomainRepository(BaseRepository):
    def create(self, domain: Domain) -> Domain:
        def _work(tx: ManagedTransaction) -> dict[str, Any]:
            result = tx.run(
                """
                CREATE (d:Domain {domain_id: $domain_id, name: $name})
                RETURN d {.*, element_id: elementId(d)} AS domain
                """,
                **domain.to_properties(),
            )
            return result.single(strict=True)["domain"]

        with self._session() as session:
            data = session.execute_write(_work)
            return Domain.from_node(data)

    def find_by_id(self, domain_id: str) -> Domain | None:
        def _work(tx: ManagedTransaction) -> dict[str, Any] | None:
            result = tx.run(
                """
                MATCH (d:Domain {domain_id: $domain_id})
                RETURN d {.*, element_id: elementId(d)} AS domain
                """,
                domain_id=domain_id,
            )
            record = result.single()
            return record["domain"] if record else None

        with self._session() as session:
            data = session.execute_read(_work)
            return Domain.from_node(data) if data else None

    def delete(self, domain_id: str) -> bool:
        def _work(tx: ManagedTransaction) -> int:
            result = tx.run(
                """
                MATCH (d:Domain {domain_id: $domain_id})
                DETACH DELETE d
                RETURN count(*) AS deleted
                """,
                domain_id=domain_id,
            )
            return result.single(strict=True)["deleted"]

        with self._session() as session:
            return session.execute_write(_work) > 0

    def add_child(self, parent_id: str, child_id: str) -> None:
        def _work(tx: ManagedTransaction) -> None:
            tx.run(
                """
                MATCH (p:Domain {domain_id: $parent_id})
                MATCH (c:Domain {domain_id: $child_id})
                MERGE (p)-[:HAS_CHILD]->(c)
                """,
                parent_id=parent_id,
                child_id=child_id,
            )

        with self._session() as session:
            session.execute_write(_work)

    def get_root_domains(self) -> list[Domain]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (d:Domain)
            WHERE NOT EXISTS { MATCH (:Domain)-[:HAS_CHILD]->(d) }
            RETURN d {.*, element_id: elementId(d)} AS domain
            ORDER BY d.name
            """,
            database_=self._database,
            routing_="r",
        )
        return [Domain.from_node(r["domain"]) for r in records]
