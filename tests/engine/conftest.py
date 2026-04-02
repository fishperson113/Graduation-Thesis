from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def clean_database():
    """Override the root-level autouse clean_database fixture.

    Engine tests are pure-math and do not require Neo4j.
    """
    yield
