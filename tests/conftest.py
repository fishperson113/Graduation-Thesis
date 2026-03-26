from __future__ import annotations

import os

import pytest
from neo4j import GraphDatabase

from student_modeling.config import Settings


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(
        neo4j_uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "changeme"),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )


@pytest.fixture(scope="session")
def neo4j_driver(settings):
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
    )
    driver.verify_connectivity()
    yield driver
    driver.close()


@pytest.fixture(autouse=True)
def clean_database(neo4j_driver, settings):
    """Clean all test data before each test."""
    with neo4j_driver.session(database=settings.neo4j_database) as session:
        session.run("MATCH (n) DETACH DELETE n").consume()
    yield


@pytest.fixture
def database_name(settings) -> str:
    return settings.neo4j_database
