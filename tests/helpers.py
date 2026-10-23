"""Shared test helpers (fake encoders, record builders, synthetic videos)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.video_processor import FrameRecord

DIM = 512


class FakeEncoder:
    """Deterministic 512-d encoder so tests do not load CLIP or torch.

    Every image maps to unit vector e0; text containing "bicycle" maps to e0,
    anything else maps to e1.
    """

    dim = DIM
    model_name = "fake-ViT-B/32"

    def __init__(self) -> None:
        self.image_calls: list[list[str]] = []

    def encode_images(self, image_paths: list[str], batch_size: int = 32) -> np.ndarray:
        self.image_calls.append(list(image_paths))
        vectors = np.zeros((len(image_paths), self.dim), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors

    def encode_text(self, query: str) -> np.ndarray:
        vector = np.zeros((1, self.dim), dtype=np.float32)
        vector[0, 0 if "bicycle" in query.lower() else 1] = 1.0
        return vector


def unit(index: int, dim: int = DIM) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[index] = 1.0
    return vector


def touch_frame(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-real-jpeg")
    return path


def make_record(
    video_id: str,
    frame_number: int,
    *,
    frame_path: Path | str | None = None,
    timestamp: float | None = None,
    video_name: str = "ride.mp4",
    video_path: Path | str | None = None,
    duration: float = 20.0,
) -> FrameRecord:
    frame_filename = f"{video_id}_f{frame_number:08d}.jpg"
    return FrameRecord(
        frame_id=f"{video_id}_{frame_number}",
        video_id=video_id,
        video_filename=video_name,
        video_path=str(video_path) if video_path is not None else f"/videos/{video_name}",
        frame_filename=frame_filename,
        frame_path=str(frame_path) if frame_path is not None else f"/processed/frames/{frame_filename}",
        frame_number=frame_number,
        timestamp_sec=frame_number / 30.0 if timestamp is None else timestamp,
        duration_sec=duration,
        fps=30.0,
        width=640,
        height=480,
        interval_sec=2.0,
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
