from __future__ import annotations

from typing import Any

from neo4j import ManagedTransaction

from student_modeling.models.knows_edge import KnowsEdge


class KnowsRepository:
    """Repository for KNOWS relationships between User and Concept.

    Not extending BaseRepository since KNOWS is a relationship, not a node.
    """

    def __init__(self, driver, database: str = "neo4j") -> None:
        self._driver = driver
        self._database = database

    def _session(self, **kwargs):
        return self._driver.session(database=self._database, **kwargs)

    def create_or_update(
        self, user_id: str, concept_id: str, props: dict[str, Any]
    ) -> None:
        def _work(tx: ManagedTransaction) -> None:
            tx.run(
                """
                MATCH (u:User {user_id: $user_id})
                MATCH (c:Concept {concept_id: $concept_id})
                MERGE (u)-[k:KNOWS]->(c)
                SET k += $props
                """,
                user_id=user_id,
                concept_id=concept_id,
                props=props,
            )

        with self._session() as session:
            session.execute_write(_work)

    def get_edge(self, user_id: str, concept_id: str) -> KnowsEdge | None:
        def _work(tx: ManagedTransaction) -> dict[str, Any] | None:
            result = tx.run(
                """
                MATCH (u:User {user_id: $user_id})-[k:KNOWS]->(c:Concept {concept_id: $concept_id})
                RETURN k {
                    .*,
                    user_id: u.user_id,
                    concept_id: c.concept_id
                } AS edge
                """,
                user_id=user_id,
                concept_id=concept_id,
            )
            record = result.single()
            return record["edge"] if record else None

        with self._session() as session:
            data = session.execute_read(_work)
            return KnowsEdge.from_record(data) if data else None

    def get_user_overlay(self, user_id: str) -> list[KnowsEdge]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (u:User {user_id: $user_id})-[k:KNOWS]->(c:Concept)
            RETURN k {
                .*,
                user_id: u.user_id,
                concept_id: c.concept_id
            } AS edge
            ORDER BY c.concept_id
            """,
            user_id=user_id,
            database_=self._database,
            routing_="r",
        )
        return [KnowsEdge.from_record(r["edge"]) for r in records]

    def get_weak_concepts(
        self, user_id: str, threshold: float = 0.8
    ) -> list[KnowsEdge]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (u:User {user_id: $user_id})-[k:KNOWS]->(c:Concept)
            WHERE k.mastery < $threshold
            RETURN k {
                .*,
                user_id: u.user_id,
                concept_id: c.concept_id
            } AS edge
            ORDER BY k.mastery ASC
            """,
            user_id=user_id,
            threshold=threshold,
            database_=self._database,
            routing_="r",
        )
        return [KnowsEdge.from_record(r["edge"]) for r in records]

    def get_mastery_map(self, user_id: str) -> dict[str, float]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (u:User {user_id: $user_id})-[k:KNOWS]->(c:Concept)
            RETURN c.concept_id AS concept_id, k.mastery AS mastery
            """,
            user_id=user_id,
            database_=self._database,
            routing_="r",
        )
        return {r["concept_id"]: r["mastery"] for r in records}
