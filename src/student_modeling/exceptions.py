from __future__ import annotations


class StudentModelingError(Exception):
    """Base exception for the student modeling application."""


class DatabaseConnectionError(StudentModelingError):
    """Raised when database connection fails."""


class EntityNotFoundError(StudentModelingError):
    """Raised when a requested entity does not exist."""

    def __init__(self, entity_type: str, entity_id: str) -> None:
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} not found: {entity_id}")


class DuplicateEntityError(StudentModelingError):
    """Raised when attempting to create an entity that already exists."""

    def __init__(self, entity_type: str, entity_id: str) -> None:
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} already exists: {entity_id}")
