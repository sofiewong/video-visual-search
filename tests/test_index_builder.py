"""Tests for FAISS index building and frame cleanup."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pytest

import src.index_builder as index_builder
from src.database import (
    get_all_frame_metadata,
    get_connection,
    mark_video_indexed,
    replace_video_metadata,
)
from src.index_builder import (
    CONFIG_FILENAME,
    EMBEDDING_FILENAME,
    INDEX_FILENAME,
    _default_encoder,
    _resolve_frame_path,
    build_index,
    load_indexable_rows,
    load_unindexed_records,
    main,
    remove_deleted_videos,
    save_index,
)
from src.metadata_manager import replace_video_records
from tests.helpers import FakeEncoder, make_record, touch_frame


def indexed_video_ids() -> dict[str, int]:
    with get_connection() as conn:
        rows = conn.execute("SELECT video_id, frame_count FROM indexed_videos").fetchall()
    return {row["video_id"]: row["frame_count"] for row in rows}


def frames_on_disk(tmp_path: Path, video_id: str, count: int) -> tuple[list[Path], list]:
    frames_dir = tmp_path / "frames"
    video_path = touch_frame(tmp_path / "videos" / f"{video_id}.mp4")
    paths = [touch_frame(frames_dir / f"{video_id}_{i}.jpg") for i in range(count)]
    records = [
        make_record(video_id, i * 60, frame_path=path, timestamp=i * 2.0, video_path=video_path)
        for i, path in enumerate(paths)
    ]
    return paths, records


# _default_encoder

def test_default_encoder_constructs_clip_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.clip_encoder as clip_encoder

    monkeypatch.setattr(clip_encoder, "CLIPEncoder", FakeEncoder)
    assert isinstance(_default_encoder(), FakeEncoder)


# _resolve_frame_path

def test_resolve_frame_path_uses_existing_path(tmp_path: Path) -> None:
    path = touch_frame(tmp_path / "a.jpg")
    assert _resolve_frame_path({"frame_path": str(path)}, None) == path


def test_resolve_frame_path_falls_back_to_frames_dir(tmp_path: Path) -> None:
    path = touch_frame(tmp_path / "frames" / "a.jpg")
    row = {"frame_path": "/old/location/a.jpg"}
    assert _resolve_frame_path(row, tmp_path / "frames") == path


def test_resolve_frame_path_uses_frame_filename(tmp_path: Path) -> None:
    path = touch_frame(tmp_path / "frames" / "b.jpg")
    row = {"frame_path": "", "frame_filename": "b.jpg"}
    assert _resolve_frame_path(row, tmp_path / "frames") == path


def test_resolve_frame_path_returns_none_when_missing(tmp_path: Path) -> None:
    row = {"frame_path": "/nope/a.jpg", "frame_filename": "a.jpg"}
    assert _resolve_frame_path(row, tmp_path) is None
    assert _resolve_frame_path({}, None) is None


# load_indexable_rows

def test_load_indexable_rows_requires_csv(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Metadata CSV"):
        load_indexable_rows(tmp_path / "missing.csv")


def test_load_indexable_rows_splits_valid_and_missing(tmp_path: Path) -> None:
    csv_path = tmp_path / "metadata.csv"
    present = touch_frame(tmp_path / "frames" / "present.jpg")
    records = [
        make_record("v", 0, frame_path=present),
        make_record("v", 60, frame_path=tmp_path / "frames" / "missing.jpg"),
    ]
    replace_video_records(csv_path, "v", records)

    rows, paths, skipped = load_indexable_rows(csv_path, tmp_path / "frames")
    assert paths == [present.resolve()]
    assert rows[0]["frame_path"] == str(present.resolve())
    assert len(skipped) == 1 and "not found" in skipped[0]


# save_index

def test_save_index_first_build_writes_everything(tmp_path: Path) -> None:
    records = [make_record("a", 0), make_record("a", 60)]
    embeddings = np.eye(2, 512, dtype=np.float64)  # also exercises float32 cast
    index = save_index(tmp_path, embeddings, records, model_name="m")

    assert index.ntotal == 2
    assert (tmp_path / INDEX_FILENAME).is_file()
    saved = np.load(tmp_path / EMBEDDING_FILENAME)
    assert saved.dtype == np.float32 and saved.shape == (2, 512)
    config = json.loads((tmp_path / CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert config["model_name"] == "m"
    assert config["embedding_dim"] == 512
    assert [r["frame_id"] for r in get_all_frame_metadata()] == ["a_0", "a_60"]
    assert indexed_video_ids() == {"a": 2}


def test_save_index_appends_to_existing_index(tmp_path: Path) -> None:
    save_index(tmp_path, np.eye(2, 512, dtype=np.float32), [make_record("a", 0), make_record("a", 60)])
    index = save_index(tmp_path, np.eye(1, 512, dtype=np.float32), [make_record("b", 0)])

    assert index.ntotal == 3
    assert np.load(tmp_path / EMBEDDING_FILENAME).shape == (3, 512)
    assert [r["frame_id"] for r in get_all_frame_metadata()] == ["a_0", "a_60", "b_0"]
    assert indexed_video_ids() == {"a": 2, "b": 1}


def test_save_index_rejects_bad_shapes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2-D"):
        save_index(tmp_path, np.zeros(512, dtype=np.float32), [make_record("a", 0)])
    with pytest.raises(ValueError, match="count"):
        save_index(tmp_path, np.zeros((2, 512), dtype=np.float32), [make_record("a", 0)])


def test_save_index_rejects_dim_mismatch_with_existing(tmp_path: Path) -> None:
    save_index(tmp_path, np.eye(1, 512, dtype=np.float32), [make_record("a", 0)])
    with pytest.raises(ValueError, match="dimension"):
        save_index(tmp_path, np.eye(1, 256, dtype=np.float32), [make_record("b", 0)])


def test_save_index_requires_existing_embeddings(tmp_path: Path) -> None:
    save_index(tmp_path, np.eye(1, 512, dtype=np.float32), [make_record("a", 0)])
    (tmp_path / EMBEDDING_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match="embeddings"):
        save_index(tmp_path, np.eye(1, 512, dtype=np.float32), [make_record("b", 0)])


def test_save_index_detects_embedding_count_mismatch(tmp_path: Path) -> None:
    save_index(tmp_path, np.eye(1, 512, dtype=np.float32), [make_record("a", 0)])
    np.save(tmp_path / EMBEDDING_FILENAME, np.zeros((5, 512), dtype=np.float32))
    with pytest.raises(ValueError, match="does not match"):
        save_index(tmp_path, np.eye(1, 512, dtype=np.float32), [make_record("b", 0)])


# build_index

def test_build_index_encodes_and_deletes_frames(tmp_path: Path) -> None:
    paths, records = frames_on_disk(tmp_path, "a", 3)
    encoder = FakeEncoder()
    count = build_index(records, index_dir=tmp_path / "index", encoder=encoder)

    assert count == 3
    assert encoder.image_calls == [[str(p) for p in paths]]
    assert (tmp_path / "index" / INDEX_FILENAME).is_file()
    assert not any(p.exists() for p in paths)


def test_build_index_can_keep_frames(tmp_path: Path) -> None:
    paths, records = frames_on_disk(tmp_path, "a", 2)
    build_index(records, index_dir=tmp_path / "index", encoder=FakeEncoder(), delete_frames=False)
    assert all(p.exists() for p in paths)


def test_build_index_keeps_frames_when_save_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, records = frames_on_disk(tmp_path, "a", 2)

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(index_builder, "save_index", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        build_index(records, index_dir=tmp_path / "index", encoder=FakeEncoder())
    assert all(p.exists() for p in paths)


def test_build_index_reports_undeletable_frames(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths, records = frames_on_disk(tmp_path, "a", 2)

    class DeletingEncoder(FakeEncoder):
        def encode_images(self, image_paths, batch_size=32):
            Path(image_paths[0]).unlink()  # simulate a file vanishing before cleanup
            return super().encode_images(image_paths, batch_size)

    assert build_index(records, index_dir=tmp_path / "index", encoder=DeletingEncoder()) == 2
    out = capsys.readouterr().out
    assert "Could not delete" in out
    assert "Deleted 1 frame images." in out


def test_build_index_skips_missing_frames_and_uses_frames_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    frames_dir = tmp_path / "frames"
    moved = touch_frame(frames_dir / "moved.jpg")
    records = [
        make_record("a", 0, frame_path="/old/place/moved.jpg"),
        make_record("a", 60, frame_path=frames_dir / "gone.jpg"),
    ]
    encoder = FakeEncoder()
    count = build_index(records, index_dir=tmp_path / "index", frames_dir=frames_dir, encoder=encoder)

    assert count == 1
    assert encoder.image_calls == [[str(moved)]]
    assert "Missing frame" in capsys.readouterr().out
    assert [r["frame_id"] for r in get_all_frame_metadata()] == ["a_0"]


def test_build_index_requires_some_frames(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No valid frame"):
        build_index([make_record("a", 0)], index_dir=tmp_path / "index", encoder=FakeEncoder())


def test_build_index_rejects_wrong_vector_count(tmp_path: Path) -> None:
    _, records = frames_on_disk(tmp_path, "a", 2)

    class ShortEncoder(FakeEncoder):
        def encode_images(self, image_paths, batch_size=32):
            return super().encode_images(image_paths[:1], batch_size)

    with pytest.raises(RuntimeError, match="different number"):
        build_index(records, index_dir=tmp_path / "index", encoder=ShortEncoder())


def test_build_index_rejects_wrong_dim(tmp_path: Path) -> None:
    _, records = frames_on_disk(tmp_path, "a", 1)

    class WrongDimEncoder(FakeEncoder):
        def encode_images(self, image_paths, batch_size=32):
            self.dim = 256  # claims 256-d after producing 512-d vectors
            return np.ones((len(image_paths), 512), dtype=np.float32)

    with pytest.raises(RuntimeError, match="dim"):
        build_index(records, index_dir=tmp_path / "index", encoder=WrongDimEncoder())


def test_build_index_uses_default_encoder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, records = frames_on_disk(tmp_path, "a", 1)
    monkeypatch.setattr(index_builder, "_default_encoder", FakeEncoder)
    assert build_index(records, index_dir=tmp_path / "index") == 1


# remove_deleted_videos

def build_three_videos(tmp_path: Path) -> Path:
    """Index videos a (2 frames), b (1 frame), c (2 frames) with distinct vectors."""
    index_dir = tmp_path / "index"
    for video_id, count in [("a", 2), ("b", 1), ("c", 2)]:
        _, records = frames_on_disk(tmp_path, video_id, count)
        replace_video_metadata(video_id, records)
        build_index(records, index_dir=index_dir, encoder=FakeEncoder())
    # give each frame a unique vector so we can check which rows survived
    np.save(index_dir / EMBEDDING_FILENAME, np.eye(5, 512, dtype=np.float32))
    return index_dir


def test_remove_deleted_videos_does_nothing_when_all_exist(tmp_path: Path) -> None:
    index_dir = build_three_videos(tmp_path)
    assert remove_deleted_videos(index_dir) == []
    assert len(get_all_frame_metadata()) == 5


def test_remove_deleted_videos_drops_vectors_and_metadata(tmp_path: Path) -> None:
    index_dir = build_three_videos(tmp_path)
    video_b = tmp_path / "videos" / "b.mp4"
    video_b.unlink()

    assert remove_deleted_videos(index_dir) == [str(video_b)]

    index = faiss.read_index(str(index_dir / INDEX_FILENAME))
    embeddings = np.load(index_dir / EMBEDDING_FILENAME)
    assert index.ntotal == 4
    np.testing.assert_array_equal(embeddings, np.eye(5, 512, dtype=np.float32)[[0, 1, 3, 4]])
    records = get_all_frame_metadata()  # also checks IDs are contiguous again
    assert [r["video_id"] for r in records] == ["a", "a", "c", "c"]
    assert indexed_video_ids() == {"a": 2, "c": 2}


def test_remove_deleted_videos_then_index_new_video(tmp_path: Path) -> None:
    index_dir = build_three_videos(tmp_path)
    (tmp_path / "videos" / "a.mp4").unlink()
    remove_deleted_videos(index_dir)

    _, records = frames_on_disk(tmp_path, "d", 1)
    replace_video_metadata("d", records)
    build_index(records, index_dir=index_dir, encoder=FakeEncoder())

    assert faiss.read_index(str(index_dir / INDEX_FILENAME)).ntotal == 4
    assert [r["video_id"] for r in get_all_frame_metadata()] == ["b", "c", "c", "d"]


def test_remove_deleted_videos_all_videos(tmp_path: Path) -> None:
    index_dir = build_three_videos(tmp_path)
    for video in (tmp_path / "videos").iterdir():
        video.unlink()
    assert len(remove_deleted_videos(index_dir)) == 3
    assert faiss.read_index(str(index_dir / INDEX_FILENAME)).ntotal == 0
    assert np.load(index_dir / EMBEDDING_FILENAME).shape == (0, 512)
    assert get_all_frame_metadata() == []


def test_remove_deleted_videos_without_index_deletes_frames(tmp_path: Path) -> None:
    paths, records = frames_on_disk(tmp_path, "a", 2)
    replace_video_metadata("a", records)
    (tmp_path / "videos" / "a.mp4").unlink()

    assert len(remove_deleted_videos(tmp_path / "index")) == 1
    assert not any(p.exists() for p in paths)
    assert load_unindexed_records() == []


def test_remove_deleted_videos_rejects_mismatched_embeddings(tmp_path: Path) -> None:
    index_dir = build_three_videos(tmp_path)
    np.save(index_dir / EMBEDDING_FILENAME, np.zeros((2, 512), dtype=np.float32))
    (tmp_path / "videos" / "b.mp4").unlink()
    with pytest.raises(ValueError, match="does not match"):
        remove_deleted_videos(index_dir)


# main

def test_main_removes_deleted_videos(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    index_dir = build_three_videos(tmp_path)
    (tmp_path / "videos" / "c.mp4").unlink()
    assert main(["--output-dir", str(tmp_path), "--index-dir", str(index_dir)]) == 0
    out = capsys.readouterr().out
    assert "Removed deleted video" in out and "c.mp4" in out
    assert faiss.read_index(str(index_dir / INDEX_FILENAME)).ntotal == 3


def test_load_unindexed_records(tmp_path: Path) -> None:
    replace_video_metadata("a", [make_record("a", 0)])
    replace_video_metadata("b", [make_record("b", 0)])
    mark_video_indexed("a", 1)
    assert load_unindexed_records() == [make_record("b", 0)]


def test_main_with_nothing_to_index(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--output-dir", str(tmp_path), "--index-dir", str(tmp_path / "index")]) == 0
    assert "already indexed" in capsys.readouterr().out


def test_main_indexes_unindexed_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, records = frames_on_disk(tmp_path, "a", 2)
    replace_video_metadata("a", records)
    monkeypatch.setattr(index_builder, "_default_encoder", FakeEncoder)

    assert main(["--output-dir", str(tmp_path), "--index-dir", str(tmp_path / "index")]) == 0
    assert (tmp_path / "index" / INDEX_FILENAME).is_file()
    assert indexed_video_ids() == {"a": 2}
    assert not any(p.exists() for p in paths)


def test_main_reports_missing_frames(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths, records = frames_on_disk(tmp_path, "a", 1)
    paths[0].unlink()
    replace_video_metadata("a", records)
    assert main(["--output-dir", str(tmp_path), "--index-dir", str(tmp_path / "index")]) == 1
    assert "No valid frame" in capsys.readouterr().err
