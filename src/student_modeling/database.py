from __future__ import annotations

import logging

import neo4j
from neo4j import AsyncGraphDatabase, GraphDatabase

from student_modeling.config import Settings

logger = logging.getLogger(__name__)


class Database:
    """Singleton sync Neo4j driver. Thread-safe; create sessions per-thread."""

    _driver: neo4j.Driver | None = None
    _database: str = "neo4j"

    @classmethod
    def connect(cls, settings: Settings) -> None:
        if cls._driver is not None:
            raise RuntimeError("Database already connected. Call close() first.")
        cls._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
        )
        cls._database = settings.neo4j_database
        cls._driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", settings.neo4j_uri)

    @classmethod
    def close(cls) -> None:
        if cls._driver is not None:
            cls._driver.close()
            cls._driver = None
            logger.info("Neo4j driver closed")

    @classmethod
    def get_driver(cls) -> neo4j.Driver:
        if cls._driver is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return cls._driver

    @classmethod
    def session(cls, **kwargs) -> neo4j.Session:
        return cls.get_driver().session(database=cls._database, **kwargs)


class AsyncDatabase:
    """Singleton async Neo4j driver."""

    _driver: neo4j.AsyncDriver | None = None
    _database: str = "neo4j"

    @classmethod
    async def connect(cls, settings: Settings) -> None:
        if cls._driver is not None:
            raise RuntimeError("AsyncDatabase already connected. Call close() first.")
        cls._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
        )
        cls._database = settings.neo4j_database
        await cls._driver.verify_connectivity()
        logger.info("Async connected to Neo4j at %s", settings.neo4j_uri)

    @classmethod
    async def close(cls) -> None:
        if cls._driver is not None:
            await cls._driver.close()
            cls._driver = None
            logger.info("Async Neo4j driver closed")

    @classmethod
    def get_driver(cls) -> neo4j.AsyncDriver:
        if cls._driver is None:
            raise RuntimeError("AsyncDatabase not connected. Call connect() first.")
        return cls._driver

    @classmethod
    def session(cls, **kwargs) -> neo4j.AsyncSession:
        return cls.get_driver().session(database=cls._database, **kwargs)
