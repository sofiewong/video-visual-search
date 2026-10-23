"""Tests for timestamps, sampling, and invalid-video handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.video_processor as video_processor
from src.video_processor import (
    VideoProcessingError,
    _write_frame,
    extract_frames,
    frame_number_to_timestamp,
    list_videos,
    make_video_id,
    process_video,
    resolve_video_targets,
    sample_frame_indices,
    validate_video,
)
from tests.helpers import write_synthetic_video


# make_video_id

def test_make_video_id_is_stable_and_sanitized(tmp_path: Path) -> None:
    path = tmp_path / "my clip (1)!.mp4"
    video_id = make_video_id(path)
    assert video_id == make_video_id(path)
    stem, digest = video_id.rsplit("_", 1)
    assert stem == "my_clip_1"
    assert len(digest) == 8


def test_make_video_id_differs_by_directory(tmp_path: Path) -> None:
    assert make_video_id(tmp_path / "a" / "x.mp4") != make_video_id(tmp_path / "b" / "x.mp4")


def test_make_video_id_falls_back_for_symbol_names(tmp_path: Path) -> None:
    assert make_video_id(tmp_path / "!!!.mp4").startswith("video_")


# list_videos

def test_list_videos_filters_unsupported(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not-a-real-video")
    upper = tmp_path / "other.MOV"
    upper.write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    (tmp_path / "dir.mp4").mkdir()
    assert list_videos(tmp_path) == [video, upper]


def test_list_videos_requires_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list_videos(tmp_path / "missing")


# frame_number_to_timestamp

def test_frame_number_to_timestamp() -> None:
    assert frame_number_to_timestamp(0, 30.0) == 0.0
    assert frame_number_to_timestamp(60, 30.0) == 2.0
    assert frame_number_to_timestamp(15, 10.0) == 1.5


def test_frame_number_to_timestamp_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        frame_number_to_timestamp(0, 0)
    with pytest.raises(ValueError):
        frame_number_to_timestamp(-1, 30.0)


# sample_frame_indices

def test_sample_frame_indices_every_two_seconds() -> None:
    assert sample_frame_indices(fps=30.0, frame_count=180, interval_sec=2.0) == [0, 60, 120]


def test_sample_frame_indices_includes_last_frame_when_on_boundary() -> None:
    assert sample_frame_indices(fps=10.0, frame_count=21, interval_sec=1.0)[-1] == 20


def test_sample_frame_indices_no_duplicates_for_tiny_interval() -> None:
    indices = sample_frame_indices(fps=2.0, frame_count=4, interval_sec=0.1)
    assert indices == [0, 1, 2, 3]


def test_sample_frame_indices_single_frame() -> None:
    assert sample_frame_indices(fps=30.0, frame_count=1, interval_sec=2.0) == [0]


@pytest.mark.parametrize("fps,count,interval", [(0, 10, 1), (30, 0, 1), (30, 10, 0)])
def test_sample_frame_indices_rejects_bad_inputs(fps: float, count: int, interval: float) -> None:
    with pytest.raises(ValueError):
        sample_frame_indices(fps, count, interval)


# validate_video

def test_validate_video_reads_properties(tmp_path: Path) -> None:
    path = tmp_path / "sample.mp4"
    write_synthetic_video(path, fps=10.0, frame_count=20, width=32, height=24)
    info = validate_video(path)
    assert info.fps == pytest.approx(10.0)
    assert info.frame_count == 20
    assert info.duration_sec == pytest.approx(2.0)
    assert (info.width, info.height) == (32, 24)
    assert info.filename == "sample.mp4"
    assert info.video_id == make_video_id(path)
    assert info.path == path.resolve()


def test_validate_video_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(VideoProcessingError, match="not found"):
        validate_video(tmp_path / "gone.mp4")


def test_validate_video_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "clip.mkv"
    path.write_bytes(b"data")
    with pytest.raises(VideoProcessingError, match="Unsupported"):
        validate_video(path)


def test_validate_video_rejects_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"this is not a video container")
    with pytest.raises(VideoProcessingError):
        validate_video(path)


# _write_frame

def test_write_frame_creates_parent_dirs(tmp_path: Path) -> None:
    destination = tmp_path / "a" / "b" / "frame.jpg"
    _write_frame(np.zeros((8, 8, 3), dtype=np.uint8), destination)
    assert destination.is_file()


def test_write_frame_raises_when_write_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_processor.cv2, "imwrite", lambda path, image: False)
    with pytest.raises(VideoProcessingError, match="Failed to write"):
        _write_frame(np.zeros((8, 8, 3), dtype=np.uint8), tmp_path / "frame.jpg")


# extract_frames

def test_extract_frames_records_match_samples(tmp_path: Path) -> None:
    path = tmp_path / "sample.mp4"
    write_synthetic_video(path, fps=10.0, frame_count=25)
    video = validate_video(path)
    records = extract_frames(video, tmp_path / "frames", interval_sec=1.0)

    assert [r.frame_number for r in records] == [0, 10, 20]
    assert [r.timestamp_sec for r in records] == pytest.approx([0.0, 1.0, 2.0])
    first = records[0]
    assert first.frame_id == f"{video.video_id}_0"
    assert first.frame_filename == f"{video.video_id}_f00000000.jpg"
    assert Path(first.frame_path).is_file()
    assert first.interval_sec == 1.0
    assert first.duration_sec == pytest.approx(2.5)


def test_extract_frames_raises_when_video_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "sample.mp4"
    write_synthetic_video(path)
    video = validate_video(path)
    path.write_bytes(b"corrupted after validation")
    with pytest.raises(VideoProcessingError):
        extract_frames(video, tmp_path / "frames", interval_sec=1.0)


# process_video

def test_process_video_from_synthetic_video(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    frames_dir = tmp_path / "frames"
    write_synthetic_video(video_path, fps=10.0, frame_count=25)

    info, records = process_video(video_path, frames_dir, interval_sec=1.0)
    assert info.fps > 0
    assert records
    assert all((frames_dir / record.frame_filename).is_file() for record in records)
    assert records[0].timestamp_sec == pytest.approx(0.0)
    assert all(record.video_id == info.video_id for record in records)
    names = [record.frame_filename for record in records]
    assert len(names) == len(set(names))


def test_process_video_propagates_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(VideoProcessingError):
        process_video(tmp_path / "missing.mp4", tmp_path / "frames", 1.0)


# resolve_video_targets

def test_resolve_video_targets_single_video(tmp_path: Path) -> None:
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    assert resolve_video_targets(tmp_path / "ignored", video) == ([video], [])


def test_resolve_video_targets_missing_video(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Video file"):
        resolve_video_targets(tmp_path, tmp_path / "missing.mp4")


def test_resolve_video_targets_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Input directory"):
        resolve_video_targets(tmp_path / "missing")


def test_resolve_video_targets_lists_and_skips(tmp_path: Path) -> None:
    input_dir = tmp_path / "videos"
    input_dir.mkdir()
    video = input_dir / "a.mp4"
    video.write_bytes(b"x")
    (input_dir / "notes.txt").write_text("x", encoding="utf-8")
    (input_dir / ".gitkeep").write_text("", encoding="utf-8")
    videos, skipped = resolve_video_targets(input_dir)
    assert videos == [video]
    assert skipped == ["notes.txt: skipped unsupported format '.txt'"]
