from __future__ import annotations

import pytest

from student_modeling.models.course import Course
from student_modeling.models.student import Student
from student_modeling.repositories.course_repository import CourseRepository
from student_modeling.repositories.student_repository import StudentRepository


@pytest.fixture
def student_repo(neo4j_driver, database_name):
    return StudentRepository(neo4j_driver, database_name)


@pytest.fixture
def course_repo(neo4j_driver, database_name):
    return CourseRepository(neo4j_driver, database_name)


@pytest.fixture
def sample_student() -> Student:
    return Student(student_id="STU-001", name="Alice Johnson", email="alice@example.com")


@pytest.fixture
def sample_course() -> Course:
    return Course(course_id="CS-101", title="Intro to CS", description="Basics", difficulty_level=1)


class TestStudentRepository:
    def test_create_and_find(self, student_repo, sample_student):
        created = student_repo.create(sample_student)
        assert created.student_id == "STU-001"
        assert created.name == "Alice Johnson"
        assert created.enrolled_at is not None

        found = student_repo.find_by_id("STU-001")
        assert found is not None
        assert found.student_id == "STU-001"

    def test_find_nonexistent_returns_none(self, student_repo):
        assert student_repo.find_by_id("NOPE") is None

    def test_find_all(self, student_repo):
        for i in range(5):
            student_repo.create(Student(f"STU-{i:03d}", f"Student {i}", f"s{i}@test.com"))
        students = student_repo.find_all(limit=3)
        assert len(students) == 3

    def test_update(self, student_repo, sample_student):
        student_repo.create(sample_student)
        sample_student.name = "Alice M. Johnson"
        updated = student_repo.update(sample_student)
        assert updated.name == "Alice M. Johnson"

    def test_delete(self, student_repo, sample_student):
        student_repo.create(sample_student)
        assert student_repo.delete("STU-001") is True
        assert student_repo.find_by_id("STU-001") is None

    def test_delete_nonexistent(self, student_repo):
        assert student_repo.delete("NOPE") is False

    def test_bulk_create(self, student_repo):
        students = [
            Student(f"BULK-{i:04d}", f"Bulk Student {i}", f"bulk{i}@test.com")
            for i in range(50)
        ]
        count = student_repo.bulk_create(students)
        assert count == 50

    def test_enroll_and_get_courses(self, student_repo, course_repo, sample_student, sample_course):
        student_repo.create(sample_student)
        course_repo.create(sample_course)
        student_repo.enroll_in_course("STU-001", "CS-101")

        courses = student_repo.get_enrolled_courses("STU-001")
        assert len(courses) == 1
        assert courses[0].course_id == "CS-101"
