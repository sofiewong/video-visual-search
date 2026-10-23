"""Tests for FAISS-backed clip search (no CLIP download)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.search as search
from src.index_builder import EMBEDDING_FILENAME, save_index
from src.search import (
    SearchHit,
    _as_float,
    _as_int,
    _default_encoder,
    format_hits,
    format_timestamp,
    load_faiss_index,
    main,
    record_to_hit,
    search_query,
)
from src.database import save_frame_metadata
from tests.helpers import FakeEncoder, make_record, unit


@pytest.fixture
def index_dir(tmp_path: Path) -> Path:
    """Index with a bicycle clip (t=0-4, e0) and a cat clip (t=20-22, e1)."""
    records = [
        make_record("ride", 0, timestamp=0.0),
        make_record("ride", 60, timestamp=2.0),
        make_record("ride", 120, timestamp=4.0),
        make_record("ride", 600, timestamp=20.0),
        make_record("ride", 660, timestamp=22.0),
    ]
    embeddings = np.stack([unit(0), unit(0), unit(0), unit(1), unit(1)])
    out = tmp_path / "index"
    save_index(out, embeddings, records)
    return out


def test_default_encoder_constructs_clip_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.clip_encoder as clip_encoder

    monkeypatch.setattr(clip_encoder, "CLIPEncoder", FakeEncoder)
    assert isinstance(_default_encoder(), FakeEncoder)


# load_faiss_index

def test_load_faiss_index(index_dir: Path) -> None:
    index, records = load_faiss_index(index_dir)
    assert index.ntotal == 5
    assert len(records) == 5


def test_load_faiss_index_requires_index(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="FAISS"):
        load_faiss_index(tmp_path)


def test_load_faiss_index_detects_metadata_mismatch(index_dir: Path) -> None:
    save_frame_metadata([make_record("ride", 0)])
    with pytest.raises(RuntimeError, match="mismatch"):
        load_faiss_index(index_dir)


# helpers

def test_as_float() -> None:
    assert _as_float("2.5") == 2.5
    assert _as_float(None) == 0.0
    assert _as_float("abc", default=-1.0) == -1.0


def test_as_int() -> None:
    assert _as_int("60") == 60
    assert _as_int("60.9") == 60
    assert _as_int(None) == 0
    assert _as_int("abc", default=-1) == -1


def test_record_to_hit() -> None:
    record = {
        "video_filename": "ride.mp4",
        "timestamp_sec": "2.0",
        "frame_id": "ride_60",
        "frame_path": "/f.jpg",
        "video_id": "ride",
        "frame_number": "60",
    }
    hit = record_to_hit(1, np.float32(0.75), record)
    assert hit == SearchHit(1, "ride.mp4", 2.0, pytest.approx(0.75), "ride_60", "/f.jpg", "ride", 60)


def test_record_to_hit_defaults_missing_fields() -> None:
    hit = record_to_hit(3, 0.1, {})
    assert hit.video_filename == "" and hit.timestamp_sec == 0.0 and hit.frame_number == 0


def test_format_timestamp() -> None:
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(65.9) == "00:01:05"
    assert format_timestamp(3725) == "01:02:05"


def test_format_hits_empty() -> None:
    assert format_hits("cat", []) == 'No matches for "cat".'


def test_format_hits_renders_clips() -> None:
    clips = [
        {"video_filename": "ride.mp4", "start_sec": 0.0, "end_sec": 5.0, "score": 0.5,
         "matched_frames": 3, "video_path": "/videos/ride.mp4"},
        {"video_filename": "b.mp4", "start_sec": 61.0, "end_sec": 63.0, "score": 0.25,
         "matched_frames": 2},
    ]
    text = format_hits("bicycle", clips)
    assert 'Top 2 matches for "bicycle":' in text
    assert "1. ride.mp4  clip=00:00:00–00:00:05  score=0.5000" in text
    assert "Duration: 5.00s  Matching frames: 3" in text
    assert "Video: /videos/ride.mp4" in text
    assert "2. b.mp4  clip=00:01:01–00:01:03" in text
    assert text.count("Video:") == 1


# search_query

def test_search_query_returns_matching_clip(index_dir: Path) -> None:
    clips = search_query("a person riding a bicycle", index_dir=index_dir, encoder=FakeEncoder())
    assert len(clips) == 1
    clip = clips[0]
    assert clip["video_filename"] == "ride.mp4"
    assert clip["start_sec"] == 0.0
    assert clip["end_sec"] == pytest.approx(5.0)
    assert clip["matched_frames"] == 3
    assert clip["score"] == pytest.approx(1.0)


def test_search_query_respects_top_k(index_dir: Path) -> None:
    class BothEncoder(FakeEncoder):
        def encode_text(self, query):
            return (unit(0) * 0.8 + unit(1) * 0.6).reshape(1, -1)

    clips = search_query("x", index_dir=index_dir, top_k=1, encoder=BothEncoder())
    assert len(clips) == 1
    assert clips[0]["score"] == pytest.approx(0.8)


def test_search_query_accepts_1d_text_vector(index_dir: Path) -> None:
    class FlatEncoder(FakeEncoder):
        def encode_text(self, query):
            return unit(1)

    clips = search_query("cat", index_dir=index_dir, encoder=FlatEncoder())
    assert clips[0]["start_sec"] == pytest.approx(19.0)


def test_search_query_uses_default_encoder(index_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search, "_default_encoder", FakeEncoder)
    assert search_query("bicycle", index_dir=index_dir)


@pytest.mark.parametrize("query", ["", "   "])
def test_search_query_rejects_empty_query(index_dir: Path, query: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        search_query(query, index_dir=index_dir, encoder=FakeEncoder())


def test_search_query_rejects_bad_top_k(index_dir: Path) -> None:
    with pytest.raises(ValueError, match="top_k"):
        search_query("cat", index_dir=index_dir, top_k=0, encoder=FakeEncoder())


def test_search_query_requires_built_index(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="FAISS"):
        search_query("a cat", index_dir=tmp_path, encoder=FakeEncoder())


def test_search_query_rejects_query_dim_mismatch(index_dir: Path) -> None:
    class SmallEncoder(FakeEncoder):
        def encode_text(self, query):
            return np.ones((1, 8), dtype=np.float32)

    with pytest.raises(RuntimeError, match="Query dim"):
        search_query("cat", index_dir=index_dir, encoder=SmallEncoder())


def test_search_query_requires_embeddings_file(index_dir: Path) -> None:
    (index_dir / EMBEDDING_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match="frame_embeddings"):
        search_query("cat", index_dir=index_dir, encoder=FakeEncoder())


def test_search_query_detects_embedding_count_mismatch(index_dir: Path) -> None:
    np.save(index_dir / EMBEDDING_FILENAME, np.zeros((2, 512), dtype=np.float32))
    with pytest.raises(RuntimeError, match="does not match metadata"):
        search_query("cat", index_dir=index_dir, encoder=FakeEncoder())


def test_search_query_detects_embedding_dim_mismatch(index_dir: Path) -> None:
    np.save(index_dir / EMBEDDING_FILENAME, np.zeros((5, 8), dtype=np.float32))
    with pytest.raises(RuntimeError, match="dimensions"):
        search_query("cat", index_dir=index_dir, encoder=FakeEncoder())


# main

def test_main_prints_results(index_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(search, "_default_encoder", FakeEncoder)
    assert main(["-q", "bicycle", "--index-dir", str(index_dir)]) == 0
    assert "ride.mp4" in capsys.readouterr().out


def test_main_prompts_for_query(index_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(search, "_default_encoder", FakeEncoder)
    monkeypatch.setattr("builtins.input", lambda prompt: "  bicycle  ")
    assert main(["--index-dir", str(index_dir)]) == 0
    assert 'matches for "bicycle"' in capsys.readouterr().out


def test_main_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-q", "cat", "--index-dir", str(tmp_path)]) == 1
    assert "Error:" in capsys.readouterr().err
