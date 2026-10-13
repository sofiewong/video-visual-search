"""Tests for timestamps, sampling, and invalid-video handling."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.video_processor import (
    VideoProcessingError,
    frame_number_to_timestamp,
    list_videos,
    process_video,
    sample_frame_indices,
    validate_video,
)


def write_synthetic_video(
    path: Path,
    *,
    fps: float = 10.0,
    frame_count: int = 30,
    width: int = 32,
    height: int = 24,
) -> None:
    """Write a tiny BGR video for tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), "OpenCV could not create a synthetic test video"
    try:
        for index in range(frame_count):
            pixel = int(255 * (index / max(frame_count - 1, 1)))
            frame = np.full((height, width, 3), pixel, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_frame_number_to_timestamp() -> None:
    assert frame_number_to_timestamp(0, 30.0) == 0.0
    assert frame_number_to_timestamp(60, 30.0) == 2.0
    assert frame_number_to_timestamp(15, 10.0) == 1.5


def test_frame_number_to_timestamp_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        frame_number_to_timestamp(0, 0)
    with pytest.raises(ValueError):
        frame_number_to_timestamp(-1, 30.0)


def test_sample_frame_indices_every_two_seconds() -> None:
    indices = sample_frame_indices(fps=30.0, frame_count=180, interval_sec=2.0)
    assert indices[0] == 0
    assert 60 in indices
    assert 120 in indices
    assert indices[-1] <= 179
    assert indices == sorted(set(indices))


def test_list_videos_filters_unsupported(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not-a-real-video")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    found = list_videos(tmp_path)
    assert found == [video]


def test_validate_video_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "gone.mp4"
    with pytest.raises(VideoProcessingError, match="not found"):
        validate_video(missing)


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


def test_extract_frames_from_synthetic_video(tmp_path: Path) -> None:
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
