from __future__ import annotations

from typing import Any

from neo4j import ManagedTransaction

from student_modeling.exceptions import EntityNotFoundError
from student_modeling.models.course import Course
from student_modeling.models.skill import Skill
from student_modeling.repositories.base import BaseRepository


class CourseRepository(BaseRepository):
    def create(self, course: Course) -> Course:
        def _work(tx: ManagedTransaction) -> dict[str, Any]:
            result = tx.run(
                """
                CREATE (c:Course {
                    course_id: $course_id,
                    title: $title,
                    description: $description,
                    difficulty_level: $difficulty_level
                })
                RETURN c {.*, element_id: elementId(c)} AS course
                """,
                **course.to_properties(),
            )
            return result.single(strict=True)["course"]

        with self._session() as session:
            data = session.execute_write(_work)
            return Course.from_node(data)

    def find_by_id(self, course_id: str) -> Course | None:
        def _work(tx: ManagedTransaction) -> dict[str, Any] | None:
            result = tx.run(
                """
                MATCH (c:Course {course_id: $course_id})
                RETURN c {.*, element_id: elementId(c)} AS course
                """,
                course_id=course_id,
            )
            record = result.single()
            return record["course"] if record else None

        with self._session() as session:
            data = session.execute_read(_work)
            return Course.from_node(data) if data else None

    def delete(self, course_id: str) -> bool:
        def _work(tx: ManagedTransaction) -> int:
            result = tx.run(
                """
                MATCH (c:Course {course_id: $course_id})
                WITH c, c.course_id AS cid
                DETACH DELETE c
                RETURN count(cid) AS deleted
                """,
                course_id=course_id,
            )
            return result.single(strict=True)["deleted"]

        with self._session() as session:
            return session.execute_write(_work) > 0

    def add_skill(self, course_id: str, skill_id: str, relationship: str = "TEACHES") -> None:
        def _work(tx: ManagedTransaction) -> None:
            query = f"""
                MATCH (c:Course {{course_id: $course_id}})
                MATCH (s:Skill {{skill_id: $skill_id}})
                MERGE (c)-[:{relationship}]->(s)
                RETURN c.course_id AS cid
            """
            result = tx.run(query, course_id=course_id, skill_id=skill_id)
            if result.single() is None:
                raise EntityNotFoundError("Course or Skill", f"{course_id}/{skill_id}")

        with self._session() as session:
            session.execute_write(_work)

    def get_skills(self, course_id: str) -> list[Skill]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (c:Course {course_id: $course_id})-[:TEACHES]->(s:Skill)
            RETURN s {.*, element_id: elementId(s)} AS skill
            ORDER BY s.name
            """,
            course_id=course_id,
            database_=self._database,
            routing_="r",
        )
        return [Skill.from_node(r["skill"]) for r in records]
