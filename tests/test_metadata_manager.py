"""Tests for frame metadata generation and CSV persistence."""

from __future__ import annotations

from pathlib import Path

from src.metadata_manager import (
    METADATA_FIELDS,
    _read_rows,
    _write_rows,
    load_records,
    metadata_path,
    record_to_row,
    replace_video_records,
)
from src.video_processor import FrameRecord


def make_record(video_id: str, frame_number: int, video_name: str = "a.mp4") -> FrameRecord:
    return FrameRecord(
        frame_id=f"{video_id}_{frame_number}",
        video_id=video_id,
        video_filename=video_name,
        video_path=f"/videos/{video_name}",
        frame_filename=f"{video_id}_f{frame_number:08d}.jpg",
        frame_path=f"/processed/frames/{video_id}_f{frame_number:08d}.jpg",
        frame_number=frame_number,
        timestamp_sec=frame_number / 30.0,
        duration_sec=10.0,
        fps=30.0,
        width=640,
        height=480,
        interval_sec=2.0,
    )


def test_record_to_row_has_required_fields() -> None:
    row = record_to_row(make_record("clip_abcd1234", 60))
    assert set(row.keys()) == set(METADATA_FIELDS)
    assert row["video_id"] == "clip_abcd1234"
    assert row["frame_number"] == 60
    assert row["timestamp_sec"] == 2.0
    assert row["width"] == 640
    assert row["height"] == 480


def test_csv_keeps_multiple_videos_consistent(tmp_path: Path) -> None:
    csv_path = tmp_path / "metadata.csv"
    first = [
        make_record("video_one_aaa", 0, "one.mp4"),
        make_record("video_one_aaa", 60, "one.mp4"),
    ]
    second = [make_record("video_two_bbb", 0, "two.mp4")]

    replace_video_records(csv_path, "video_one_aaa", first)
    replace_video_records(csv_path, "video_two_bbb", second)

    rows = load_records(csv_path)
    assert len(rows) == 3
    assert {row["video_id"] for row in rows} == {"video_one_aaa", "video_two_bbb"}
    assert all(field in rows[0] for field in METADATA_FIELDS)


def test_reprocessing_replaces_rows_for_same_video(tmp_path: Path) -> None:
    csv_path = tmp_path / "metadata.csv"
    original = [make_record("video_one_aaa", 0), make_record("video_one_aaa", 60)]
    other = [make_record("video_two_bbb", 0, "two.mp4")]
    replace_video_records(csv_path, "video_one_aaa", original)
    replace_video_records(csv_path, "video_two_bbb", other)

    updated = [make_record("video_one_aaa", 0), make_record("video_one_aaa", 30)]
    replace_video_records(csv_path, "video_one_aaa", updated)

    rows = load_records(csv_path)
    video_one = [row for row in rows if row["video_id"] == "video_one_aaa"]
    video_two = [row for row in rows if row["video_id"] == "video_two_bbb"]
    assert {row["frame_number"] for row in video_one} == {"0", "30"}
    assert len(video_two) == 1


def test_metadata_path() -> None:
    assert metadata_path(Path("out")) == Path("out") / "metadata.csv"


def test_read_rows_missing_file_is_empty(tmp_path: Path) -> None:
    assert _read_rows(tmp_path / "missing.csv") == []
    assert load_records(tmp_path / "missing.csv") == []


def test_write_rows_creates_dirs_and_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "nested" / "metadata.csv"
    row = {**record_to_row(make_record("v", 0)), "extra": "dropped"}
    _write_rows(csv_path, [row])
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(METADATA_FIELDS)
    [loaded] = _read_rows(csv_path)
    assert "extra" not in loaded
    assert loaded["frame_id"] == "v_0"


def test_write_rows_empty_writes_header_only(tmp_path: Path) -> None:
    csv_path = tmp_path / "metadata.csv"
    _write_rows(csv_path, [])
    assert _read_rows(csv_path) == []
    assert csv_path.read_text(encoding="utf-8").strip() == ",".join(METADATA_FIELDS)
