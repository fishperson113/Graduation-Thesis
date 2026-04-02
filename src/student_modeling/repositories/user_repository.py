from __future__ import annotations

from typing import Any

from neo4j import ManagedTransaction

from student_modeling.models.user import User
from student_modeling.repositories.base import BaseRepository


class UserRepository(BaseRepository):
    def create(self, user: User) -> User:
        def _work(tx: ManagedTransaction) -> dict[str, Any]:
            result = tx.run(
                """
                CREATE (u:User {
                    user_id: $user_id,
                    name: $name,
                    email: $email,
                    created_at: datetime()
                })
                RETURN u {.*, element_id: elementId(u)} AS user
                """,
                **user.to_properties(),
            )
            return result.single(strict=True)["user"]

        with self._session() as session:
            data = session.execute_write(_work)
            return User.from_node(data)

    def find_by_id(self, user_id: str) -> User | None:
        def _work(tx: ManagedTransaction) -> dict[str, Any] | None:
            result = tx.run(
                """
                MATCH (u:User {user_id: $user_id})
                RETURN u {.*, element_id: elementId(u)} AS user
                """,
                user_id=user_id,
            )
            record = result.single()
            return record["user"] if record else None

        with self._session() as session:
            data = session.execute_read(_work)
            return User.from_node(data) if data else None

    def delete(self, user_id: str) -> bool:
        def _work(tx: ManagedTransaction) -> int:
            result = tx.run(
                """
                MATCH (u:User {user_id: $user_id})
                DETACH DELETE u
                RETURN count(*) AS deleted
                """,
                user_id=user_id,
            )
            return result.single(strict=True)["deleted"]

        with self._session() as session:
            return session.execute_write(_work) > 0
