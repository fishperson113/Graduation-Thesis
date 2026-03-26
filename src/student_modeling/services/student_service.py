from __future__ import annotations

import logging

import neo4j

from student_modeling.models.knowledge_state import KnowledgeState
from student_modeling.models.skill import Skill
from student_modeling.models.student import Student
from student_modeling.repositories.knowledge_state_repository import KnowledgeStateRepository
from student_modeling.repositories.student_repository import StudentRepository

logger = logging.getLogger(__name__)


class StudentService:
    """Business logic for student modeling operations."""

    def __init__(self, driver: neo4j.Driver, database: str = "neo4j") -> None:
        self._student_repo = StudentRepository(driver, database)
        self._knowledge_repo = KnowledgeStateRepository(driver, database)
        self._driver = driver
        self._database = database

    def register_student(self, student_id: str, name: str, email: str) -> Student:
        student = Student(student_id=student_id, name=name, email=email)
        created = self._student_repo.create(student)
        logger.info("Registered student: %s", created.student_id)
        return created

    def enroll_student(self, student_id: str, course_id: str) -> None:
        self._student_repo.enroll_in_course(student_id, course_id)
        logger.info("Enrolled %s in %s", student_id, course_id)

    def assess_skill(
        self,
        student_id: str,
        skill_id: str,
        score: float,
    ) -> KnowledgeState:
        """Update a student's mastery based on an assessment score (0.0-1.0).

        Confidence increases with each assessment attempt.
        """
        existing = self._knowledge_repo.get_student_knowledge(student_id)
        current = next((k for k in existing if k.skill_id == skill_id), None)

        if current:
            # Weighted average: new assessment has 40% weight
            mastery = current.mastery_level * 0.6 + score * 0.4
            confidence = min(1.0, current.confidence + 0.1)
        else:
            mastery = score
            confidence = 0.3

        state = self._knowledge_repo.update_mastery(
            student_id=student_id,
            skill_id=skill_id,
            mastery_level=round(mastery, 4),
            confidence=round(confidence, 4),
        )
        logger.info(
            "Assessed %s on %s: mastery=%.2f confidence=%.2f",
            student_id, skill_id, state.mastery_level, state.confidence,
        )
        return state

    def get_learning_path(self, student_id: str) -> list[Skill]:
        """Find skills the student should learn next.

        Returns unmastered skills from enrolled courses, ordered by prerequisite
        chain (skills with satisfied prerequisites first).
        """
        records, _, _ = self._driver.execute_query(
            """
            MATCH (s:Student {student_id: $student_id})-[:ENROLLED_IN]->(c:Course)
            MATCH (c)-[:TEACHES]->(sk:Skill)
            WHERE NOT EXISTS {
                MATCH (s)-[r:HAS_KNOWLEDGE]->(sk)
                WHERE r.mastery_level >= 0.8
            }
            OPTIONAL MATCH (prereq:Skill)-[:PREREQUISITE_OF]->(sk)
            OPTIONAL MATCH (s)-[pr:HAS_KNOWLEDGE]->(prereq)
            WITH sk,
                 collect(prereq.skill_id) AS prereq_ids,
                 collect(CASE WHEN pr.mastery_level >= 0.8 THEN 1 ELSE 0 END) AS prereq_met
            WITH sk,
                 size(prereq_ids) AS total_prereqs,
                 reduce(s = 0, x IN prereq_met | s + x) AS met_prereqs
            WHERE total_prereqs = 0 OR met_prereqs = total_prereqs
            RETURN sk {.*, element_id: elementId(sk)} AS skill
            ORDER BY sk.name
            """,
            student_id=student_id,
            database_=self._database,
            routing_="r",
        )
        return [Skill.from_node(r["skill"]) for r in records]
