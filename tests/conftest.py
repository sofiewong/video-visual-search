from __future__ import annotations

from pathlib import Path

import pytest

import src.database as database


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the SQLite metadata DB at a temp file so tests never touch real data."""
    db_path = tmp_path / "test_metadata.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.initialize_database()
    return db_path
