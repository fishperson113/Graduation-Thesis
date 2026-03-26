from __future__ import annotations

import pytest

from student_modeling.config import Settings
from student_modeling.database import Database


class TestDatabase:
    def setup_method(self):
        Database._driver = None

    def teardown_method(self):
        Database.close()

    def test_connect_and_close(self, settings: Settings):
        Database.connect(settings)
        assert Database.get_driver() is not None
        Database.close()
        with pytest.raises(RuntimeError, match="not connected"):
            Database.get_driver()

    def test_double_connect_raises(self, settings: Settings):
        Database.connect(settings)
        with pytest.raises(RuntimeError, match="already connected"):
            Database.connect(settings)

    def test_get_driver_before_connect_raises(self):
        with pytest.raises(RuntimeError, match="not connected"):
            Database.get_driver()

    def test_session_returns_session(self, settings: Settings):
        Database.connect(settings)
        with Database.session() as session:
            result = session.run("RETURN 1 AS n")
            assert result.single(strict=True)["n"] == 1
