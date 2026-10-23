"""Tests for the SQLite frame metadata store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import src.database as database
from src.database import (
    delete_videos,
    get_all_frame_metadata,
    get_connection,
    get_frame_paths,
    get_frame_video_ids,
    get_video_paths,
    initialize_database,
    is_video_indexed,
    load_unindexed_frame_records,
    mark_video_indexed,
    record_to_dict,
    replace_video_metadata,
    save_frame_metadata,
)
from src.video_processor import FrameRecord
from tests.helpers import make_record


def table_names() -> set[str]:
    with get_connection() as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {row["name"] for row in rows}


def test_get_connection_creates_parent_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nested" / "dir" / "meta.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    conn = get_connection()
    try:
        assert db_path.parent.is_dir()
        assert conn.row_factory is sqlite3.Row
    finally:
        conn.close()


def test_initialize_database_creates_tables_and_is_idempotent() -> None:
    initialize_database()
    initialize_database()
    assert {"frames", "indexed_videos"} <= table_names()


def test_record_to_dict_accepts_dataclass_and_dict() -> None:
    record = make_record("vid", 0)
    assert record_to_dict(record)["video_id"] == "vid"
    data = {"video_id": "x"}
    assert record_to_dict(data) is data


def test_record_to_dict_rejects_other_types() -> None:
    with pytest.raises(TypeError):
        record_to_dict(["not", "a", "record"])
    with pytest.raises(TypeError):
        record_to_dict(FrameRecord)  # the class itself, not an instance


def test_save_frame_metadata_replace_and_append() -> None:
    save_frame_metadata([make_record("a", 0), make_record("a", 60)])
    save_frame_metadata([make_record("b", 0)], starting_id=2)
    records = get_all_frame_metadata()
    assert [r["frame_id"] for r in records] == ["a_0", "a_60", "b_0"]

    save_frame_metadata([make_record("c", 0)])
    assert [r["frame_id"] for r in get_all_frame_metadata()] == ["c_0"]


def test_save_frame_metadata_replaces_extracted_rows() -> None:
    save_frame_metadata([make_record("a", 0)])
    # extraction writes rows for a new video before it is indexed
    replace_video_metadata("b", [make_record("b", 0), make_record("b", 60)])
    save_frame_metadata([make_record("b", 0), make_record("b", 60)], starting_id=1)
    assert [r["frame_id"] for r in get_all_frame_metadata()] == ["a_0", "b_0", "b_60"]


def test_get_all_frame_metadata_round_trips_fields() -> None:
    record = make_record("a", 30)
    save_frame_metadata([record])
    [loaded] = get_all_frame_metadata()
    assert loaded == record_to_dict(record)


def test_get_all_frame_metadata_empty() -> None:
    assert get_all_frame_metadata() == []


def test_get_all_frame_metadata_rejects_non_contiguous_ids() -> None:
    save_frame_metadata([make_record("a", 0), make_record("a", 60)])
    with get_connection() as conn:
        conn.execute("DELETE FROM frames WHERE faiss_id = 0")
    with pytest.raises(RuntimeError, match="contiguous"):
        get_all_frame_metadata()


def test_replace_video_metadata_only_touches_one_video() -> None:
    replace_video_metadata("a", [make_record("a", 0), make_record("a", 60)])
    replace_video_metadata("b", [make_record("b", 0)])
    replace_video_metadata("a", [make_record("a", 30)])
    with get_connection() as conn:
        rows = conn.execute("SELECT video_id, frame_number FROM frames").fetchall()
    assert sorted((r["video_id"], r["frame_number"]) for r in rows) == [("a", 30), ("b", 0)]


def test_replace_video_metadata_rejects_indexed_video() -> None:
    save_frame_metadata([make_record("a", 0)])
    mark_video_indexed("a", 1)
    with pytest.raises(ValueError, match="already indexed"):
        replace_video_metadata("a", [make_record("a", 30)])
    assert [r["frame_id"] for r in get_all_frame_metadata()] == ["a_0"]


def test_get_video_paths() -> None:
    replace_video_metadata("a", [make_record("a", 0, video_name="a.mp4"), make_record("a", 60, video_name="a.mp4")])
    replace_video_metadata("b", [make_record("b", 0, video_name="b.mp4")])
    assert get_video_paths() == {"a": "/videos/a.mp4", "b": "/videos/b.mp4"}


def test_get_frame_video_ids() -> None:
    save_frame_metadata([make_record("a", 0), make_record("b", 0), make_record("a", 60)])
    assert get_frame_video_ids() == ["a", "b", "a"]


def test_delete_videos_renumbers_ids() -> None:
    save_frame_metadata([make_record("a", 0), make_record("b", 0), make_record("b", 60), make_record("c", 0)])
    mark_video_indexed("a", 1)
    mark_video_indexed("b", 2)

    frame_paths = delete_videos({"b"})
    assert sorted(frame_paths) == [make_record("b", 0).frame_path, make_record("b", 60).frame_path]
    assert [r["frame_id"] for r in get_all_frame_metadata()] == ["a_0", "c_0"]
    assert is_video_indexed("a") and not is_video_indexed("b")


def test_delete_videos_empty_set_is_noop() -> None:
    save_frame_metadata([make_record("a", 0)])
    assert delete_videos(set()) == []
    assert len(get_all_frame_metadata()) == 1


def test_get_frame_paths() -> None:
    replace_video_metadata("a", [make_record("a", 0), make_record("a", 60)])
    replace_video_metadata("b", [make_record("b", 0)])
    assert get_frame_paths("a") == [make_record("a", 0).frame_path, make_record("a", 60).frame_path]
    assert get_frame_paths("missing") == []


def test_is_video_indexed() -> None:
    assert not is_video_indexed("a")
    mark_video_indexed("a", 1)
    assert is_video_indexed("a")


def test_load_unindexed_frame_records_skips_indexed_videos() -> None:
    replace_video_metadata("a", [make_record("a", 0)])
    replace_video_metadata("b", [make_record("b", 0), make_record("b", 60)])
    mark_video_indexed("a", 1)
    items = load_unindexed_frame_records()
    assert [item["video_id"] for item in items] == ["b", "b"]
    assert items[0]["record"]["frame_id"] == "b_0"
    assert "faiss_id" in items[0]


def test_mark_video_indexed_inserts_and_updates() -> None:
    mark_video_indexed("a", 3)
    mark_video_indexed("a", 5)
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM indexed_videos").fetchall()
    assert len(rows) == 1
    assert rows[0]["frame_count"] == 5
    assert rows[0]["indexed_at"]
