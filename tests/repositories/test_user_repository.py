from __future__ import annotations

import pytest

from student_modeling.models.user import User
from student_modeling.repositories.user_repository import UserRepository


class TestUserRepository:
    @pytest.fixture
    def repo(self, neo4j_driver, database_name):
        return UserRepository(neo4j_driver, database_name)

    def test_create_and_find(self, repo):
        user = User(user_id="u1", name="Alice", email="alice@test.com")
        created = repo.create(user)
        assert created.user_id == "u1"
        assert created.name == "Alice"

        found = repo.find_by_id("u1")
        assert found is not None
        assert found.user_id == "u1"

    def test_find_nonexistent(self, repo):
        assert repo.find_by_id("nonexistent") is None

    def test_delete(self, repo):
        user = User(user_id="u2", name="Bob", email="bob@test.com")
        repo.create(user)
        assert repo.delete("u2") is True
        assert repo.find_by_id("u2") is None

    def test_delete_nonexistent(self, repo):
        assert repo.delete("nonexistent") is False
