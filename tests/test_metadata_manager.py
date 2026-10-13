"""Tests for frame metadata generation and CSV persistence."""

from __future__ import annotations

from pathlib import Path

from src.metadata_manager import (
    METADATA_FIELDS,
    load_records,
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
