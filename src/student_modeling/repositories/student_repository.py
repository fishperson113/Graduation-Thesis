from __future__ import annotations

from typing import Any

from neo4j import ManagedTransaction

from student_modeling.exceptions import EntityNotFoundError
from student_modeling.models.course import Course
from student_modeling.models.student import Student
from student_modeling.repositories.base import BaseRepository


class StudentRepository(BaseRepository):
    def create(self, student: Student) -> Student:
        def _work(tx: ManagedTransaction) -> dict[str, Any]:
            result = tx.run(
                """
                CREATE (s:Student {
                    student_id: $student_id,
                    name: $name,
                    email: $email,
                    enrolled_at: datetime()
                })
                RETURN s {.*, element_id: elementId(s)} AS student
                """,
                **student.to_properties(),
            )
            return result.single(strict=True)["student"]

        with self._session() as session:
            data = session.execute_write(_work)
            return Student.from_node(data)

    def find_by_id(self, student_id: str) -> Student | None:
        def _work(tx: ManagedTransaction) -> dict[str, Any] | None:
            result = tx.run(
                """
                MATCH (s:Student {student_id: $student_id})
                RETURN s {.*, element_id: elementId(s)} AS student
                """,
                student_id=student_id,
            )
            record = result.single()
            return record["student"] if record else None

        with self._session() as session:
            data = session.execute_read(_work)
            return Student.from_node(data) if data else None

    def find_all(self, skip: int = 0, limit: int = 100) -> list[Student]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (s:Student)
            RETURN s {.*, element_id: elementId(s)} AS student
            ORDER BY s.name
            SKIP $skip LIMIT $limit
            """,
            skip=skip,
            limit=limit,
            database_=self._database,
            routing_="r",
        )
        return [Student.from_node(r["student"]) for r in records]

    def update(self, student: Student) -> Student:
        def _work(tx: ManagedTransaction) -> dict[str, Any]:
            result = tx.run(
                """
                MATCH (s:Student {student_id: $student_id})
                SET s.name = $name, s.email = $email
                RETURN s {.*, element_id: elementId(s)} AS student
                """,
                **student.to_properties(),
            )
            record = result.single()
            if record is None:
                raise EntityNotFoundError("Student", student.student_id)
            return record["student"]

        with self._session() as session:
            data = session.execute_write(_work)
            return Student.from_node(data)

    def delete(self, student_id: str) -> bool:
        def _work(tx: ManagedTransaction) -> int:
            result = tx.run(
                """
                MATCH (s:Student {student_id: $student_id})
                WITH s, s.student_id AS sid
                DETACH DELETE s
                RETURN count(sid) AS deleted
                """,
                student_id=student_id,
            )
            return result.single(strict=True)["deleted"]

        with self._session() as session:
            return session.execute_write(_work) > 0

    def enroll_in_course(self, student_id: str, course_id: str) -> None:
        def _work(tx: ManagedTransaction) -> None:
            result = tx.run(
                """
                MATCH (s:Student {student_id: $student_id})
                MATCH (c:Course {course_id: $course_id})
                MERGE (s)-[r:ENROLLED_IN]->(c)
                ON CREATE SET r.enrolled_at = datetime()
                RETURN s.student_id AS sid
                """,
                student_id=student_id,
                course_id=course_id,
            )
            if result.single() is None:
                raise EntityNotFoundError("Student or Course", f"{student_id}/{course_id}")

        with self._session() as session:
            session.execute_write(_work)

    def get_enrolled_courses(self, student_id: str) -> list[Course]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (s:Student {student_id: $student_id})-[:ENROLLED_IN]->(c:Course)
            RETURN c {.*, element_id: elementId(c)} AS course
            ORDER BY c.title
            """,
            student_id=student_id,
            database_=self._database,
            routing_="r",
        )
        return [Course.from_node(r["course"]) for r in records]

    def bulk_create(self, students: list[Student]) -> int:
        """Batch insert using UNWIND for efficient bulk operations."""

        def _work(tx: ManagedTransaction) -> int:
            result = tx.run(
                """
                UNWIND $students AS s
                CREATE (n:Student {
                    student_id: s.student_id,
                    name: s.name,
                    email: s.email,
                    enrolled_at: datetime()
                })
                RETURN count(n) AS created
                """,
                students=[s.to_properties() for s in students],
            )
            return result.single(strict=True)["created"]

        with self._session() as session:
            return session.execute_write(_work)
