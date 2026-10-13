"""Tests for FAISS indexing and CLIP-style query search (no CLIP download)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.index_builder import build_index, save_index
from src.metadata_manager import METADATA_FIELDS, replace_video_records
from src.search import format_hits, search_query
from src.video_processor import FrameRecord


class FakeEncoder:
    """Deterministic 512-d encoder so tests do not load CLIP or torch."""

    dim = 512
    model_name = "fake-ViT-B/32"

    def encode_images(self, image_paths: list[str], batch_size: int = 32) -> np.ndarray:
        vectors = np.zeros((len(image_paths), self.dim), dtype=np.float32)
        for i, _ in enumerate(image_paths):
            vectors[i, i % self.dim] = 1.0
        return vectors

    def encode_text(self, query: str) -> np.ndarray:
        vector = np.zeros((1, self.dim), dtype=np.float32)
        if "bicycle" in query.lower():
            vector[0, 1] = 1.0
        else:
            vector[0, 0] = 1.0
        return vector


def _touch_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-real-jpeg")


def _record(video_id: str, frame_number: int, frame_path: Path, timestamp: float) -> FrameRecord:
    return FrameRecord(
        frame_id=f"{video_id}_{frame_number}",
        video_id=video_id,
        video_filename="ride.mp4",
        video_path=str(frame_path.parent.parent / "ride.mp4"),
        frame_filename=frame_path.name,
        frame_path=str(frame_path),
        frame_number=frame_number,
        timestamp_sec=timestamp,
        duration_sec=20.0,
        fps=30.0,
        width=64,
        height=64,
        interval_sec=2.0,
    )


def test_faiss_search_returns_filename_timestamp_and_score(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    csv_path = tmp_path / "metadata.csv"
    index_dir = tmp_path / "index"
    paths = [
        frames_dir / "a.jpg",
        frames_dir / "b.jpg",
        frames_dir / "c.jpg",
    ]
    for path in paths:
        _touch_jpeg(path)
    records = [
        _record("vid_aaa", 0, paths[0], 0.0),
        _record("vid_aaa", 60, paths[1], 2.0),
        _record("vid_aaa", 120, paths[2], 4.0),
    ]
    replace_video_records(csv_path, "vid_aaa", records)
    assert set(METADATA_FIELDS).issuperset({"video_filename", "timestamp_sec"})

    n = build_index(
        metadata_csv=csv_path,
        index_dir=index_dir,
        frames_dir=frames_dir,
        encoder=FakeEncoder(),
    )
    assert n == 3
    assert (index_dir / "frames.faiss").is_file()

    hits = search_query(
        "a person riding a bicycle",
        index_dir=index_dir,
        top_k=2,
        encoder=FakeEncoder(),
    )
    assert len(hits) == 2
    top = hits[0]
    assert top.video_filename == "ride.mp4"
    assert top.timestamp_sec == pytest.approx(2.0)
    assert top.similarity == pytest.approx(1.0)
    assert top.rank == 1
    rendered = format_hits("a person riding a bicycle", hits)
    assert "ride.mp4" in rendered
    assert "t=2.00s" in rendered
    assert "score=" in rendered


def test_search_rejects_empty_query(tmp_path: Path) -> None:
    embeddings = np.eye(2, 512, dtype=np.float32)
    records = [{"video_filename": "a.mp4", "timestamp_sec": 0, "frame_id": "a_0"}]
    records.append({"video_filename": "a.mp4", "timestamp_sec": 2, "frame_id": "a_1"})
    save_index(tmp_path, embeddings, records)
    with pytest.raises(ValueError, match="non-empty"):
        search_query("  ", index_dir=tmp_path, encoder=FakeEncoder())


def test_search_requires_built_index(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="FAISS"):
        search_query("a cat", index_dir=tmp_path, encoder=FakeEncoder())
