from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import neo4j


class BaseRepository(ABC):
    """Abstract base for all Neo4j repositories.

    Each repository owns queries for one node label. The driver is thread-safe
    and shared; sessions are created per-method call.
    """

    def __init__(self, driver: neo4j.Driver, database: str = "neo4j") -> None:
        self._driver = driver
        self._database = database

    def _session(self, **kwargs) -> neo4j.Session:
        return self._driver.session(database=self._database, **kwargs)

    @abstractmethod
    def create(self, entity: Any) -> Any: ...

    @abstractmethod
    def find_by_id(self, entity_id: str) -> Any | None: ...

    @abstractmethod
    def delete(self, entity_id: str) -> bool: ...
