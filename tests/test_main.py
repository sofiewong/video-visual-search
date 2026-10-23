"""Tests for the extract / index / search CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

import src.index_builder as index_builder
import src.search as search
from src.database import get_connection, load_unindexed_frame_records
from src.main import (
    _normalize_argv,
    main,
    parse_args,
    run,
    run_extract,
    run_index,
    run_search,
    validate_extract_args,
)
from tests.helpers import FakeEncoder, write_synthetic_video


@pytest.fixture(autouse=True)
def fake_encoders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(index_builder, "_default_encoder", FakeEncoder)
    monkeypatch.setattr(search, "_default_encoder", FakeEncoder)


@pytest.fixture
def videos_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "videos"
    write_synthetic_video(directory / "one.mp4", fps=10.0, frame_count=60)
    return directory


def frame_files(output_dir: Path) -> list[Path]:
    return sorted((output_dir / "frames").glob("*.jpg"))


def frame_row_count() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]


# _normalize_argv

def test_normalize_argv() -> None:
    assert _normalize_argv([]) == []
    assert _normalize_argv(["index"]) == ["index"]
    assert _normalize_argv(["--help"]) == ["--help"]
    assert _normalize_argv(["--interval", "1"]) == ["extract", "--interval", "1"]


def test_normalize_argv_defaults_to_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "search", "-q", "cat"])
    assert _normalize_argv(None) == ["search", "-q", "cat"]


# parse_args

def test_parse_args_defaults() -> None:
    extract = parse_args(["extract"])
    assert extract.interval == 2.0 and extract.video is None
    assert extract.input_dir == Path("data/videos")

    index = parse_args(["index"])
    assert index.batch_size == 32 and index.index_dir == Path("data/index")

    search_args = parse_args(["search", "-q", "cat", "--top-k", "3"])
    assert search_args.query == "cat" and search_args.top_k == 3


def test_parse_args_legacy_flags_mean_extract() -> None:
    args = parse_args(["--interval", "1.5"])
    assert args.command == "extract" and args.interval == 1.5


def test_parse_args_without_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args([])
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


# validate_extract_args

def extract_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    values = {"interval": 2.0, "video": None, "input_dir": tmp_path, "output_dir": tmp_path / "out"}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_validate_extract_args_accepts_valid(tmp_path: Path) -> None:
    validate_extract_args(extract_args(tmp_path))


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"interval": 0}, "--interval"),
        ({"video": Path("missing.mp4")}, "Video file does not exist"),
        ({"input_dir": Path("missing_dir")}, "Input directory does not exist"),
    ],
)
def test_validate_extract_args_rejects(tmp_path: Path, overrides: dict, message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        validate_extract_args(extract_args(tmp_path, **overrides))


def test_validate_extract_args_rejects_file_as_input_dir(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit, match="not a directory"):
        validate_extract_args(extract_args(tmp_path, input_dir=file_path))


# run_extract

def test_run_extract_writes_frames_and_metadata(tmp_path: Path, videos_dir: Path) -> None:
    out = tmp_path / "out"
    (videos_dir / "notes.txt").write_text("x", encoding="utf-8")
    args = parse_args(["extract", "--input-dir", str(videos_dir), "--output-dir", str(out), "--interval", "2"])
    assert run_extract(args) == 0
    assert len(frame_files(out)) == 3
    assert frame_row_count() == 3


def test_run_extract_single_video(tmp_path: Path, videos_dir: Path) -> None:
    out = tmp_path / "out"
    args = parse_args(["extract", "--video", str(videos_dir / "one.mp4"), "--output-dir", str(out)])
    assert run_extract(args) == 0
    assert frame_row_count() == 3


def test_run_extract_skips_already_extracted(tmp_path: Path, videos_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    args = parse_args(["extract", "--input-dir", str(videos_dir), "--output-dir", str(out)])
    assert run_extract(args) == 0
    before = {path: path.stat().st_mtime_ns for path in frame_files(out)}
    capsys.readouterr()

    assert run_extract(args) == 0
    output = capsys.readouterr().out
    assert "Skipped: already extracted" in output
    assert "Already extracted: 1" in output
    assert "Frames extracted: 0" in output
    assert {path: path.stat().st_mtime_ns for path in frame_files(out)} == before
    assert frame_row_count() == 3


def test_run_extract_reextracts_when_frames_missing(tmp_path: Path, videos_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    args = parse_args(["extract", "--input-dir", str(videos_dir), "--output-dir", str(out)])
    assert run_extract(args) == 0
    frame_files(out)[0].unlink()
    capsys.readouterr()

    assert run_extract(args) == 0
    assert "Frames extracted: 3" in capsys.readouterr().out
    assert len(frame_files(out)) == 3
    assert frame_row_count() == 3


def test_run_extract_only_extracts_new_videos(tmp_path: Path, videos_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    args = parse_args(["extract", "--input-dir", str(videos_dir), "--output-dir", str(out)])
    assert run_extract(args) == 0
    write_synthetic_video(videos_dir / "two.mp4", fps=10.0, frame_count=60)
    capsys.readouterr()

    assert run_extract(args) == 0
    output = capsys.readouterr().out
    assert "Videos processed: 1" in output
    assert "Already extracted: 1" in output
    assert frame_row_count() == 6


def test_run_extract_no_videos(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    args = parse_args(["extract", "--input-dir", str(empty), "--output-dir", str(tmp_path / "out")])
    assert run_extract(args) == 1
    assert "No supported videos" in capsys.readouterr().err


def test_run_extract_all_videos_fail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "broken.mp4").write_bytes(b"not a video")
    args = parse_args(["extract", "--input-dir", str(videos), "--output-dir", str(tmp_path / "out")])
    assert run_extract(args) == 1
    assert "Failed" in capsys.readouterr().out


# run_index

def index_args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    return parse_args(["index", "--output-dir", str(tmp_path / "out"), "--index-dir", str(tmp_path / "index"), *extra])


def test_run_index_rejects_bad_batch_size(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="batch-size"):
        run_index(index_args(tmp_path, "--batch-size", "0"))


def test_run_index_nothing_to_index(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_index(index_args(tmp_path)) == 0
    assert "already indexed" in capsys.readouterr().out


def test_run_index_indexes_and_deletes_frames(tmp_path: Path, videos_dir: Path) -> None:
    out = tmp_path / "out"
    run_extract(parse_args(["extract", "--input-dir", str(videos_dir), "--output-dir", str(out)]))
    assert frame_files(out)

    assert run_index(index_args(tmp_path)) == 0
    assert (tmp_path / "index" / "frames.faiss").is_file()
    assert frame_files(out) == []
    assert load_unindexed_frame_records() == []


def test_run_index_reports_missing_frames(tmp_path: Path, videos_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    run_extract(parse_args(["extract", "--input-dir", str(videos_dir), "--output-dir", str(out)]))
    for path in frame_files(out):
        path.unlink()
    assert run_index(index_args(tmp_path)) == 1
    assert "No valid frame" in capsys.readouterr().err


# run_search

def test_run_search_rejects_bad_top_k(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="top-k"):
        run_search(parse_args(["search", "-q", "cat", "--top-k", "0", "--index-dir", str(tmp_path)]))


def test_run_search_reports_missing_index(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_search(parse_args(["search", "-q", "cat", "--index-dir", str(tmp_path)])) == 1
    assert "FAISS index not found" in capsys.readouterr().err


def test_run_search_prompts_for_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: "cat")
    assert run_search(parse_args(["search", "--index-dir", str(tmp_path)])) == 1


# run / main

def test_run_rejects_unknown_command() -> None:
    with pytest.raises(SystemExit, match="Unknown command"):
        run(argparse.Namespace(command="bogus"))


def test_main_end_to_end(tmp_path: Path, videos_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out, index = str(tmp_path / "out"), str(tmp_path / "index")
    assert main(["extract", "--input-dir", str(videos_dir), "--output-dir", out]) == 0
    assert main(["index", "--output-dir", out, "--index-dir", index]) == 0
    capsys.readouterr()

    assert main(["search", "-q", "a bicycle", "--index-dir", index]) == 0
    output = capsys.readouterr().out
    assert "one.mp4" in output
    assert "clip=00:00:00–00:00:05" in output


def test_main_reextracting_indexed_video_keeps_search_working(tmp_path: Path, videos_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out, index = str(tmp_path / "out"), str(tmp_path / "index")
    assert main(["extract", "--input-dir", str(videos_dir), "--output-dir", out]) == 0
    assert main(["index", "--output-dir", out, "--index-dir", index]) == 0
    capsys.readouterr()

    assert main(["extract", "--input-dir", str(videos_dir), "--output-dir", out]) == 0
    output = capsys.readouterr().out
    assert "Skipped: already indexed" in output
    assert "Already indexed:  1" in output
    assert frame_files(tmp_path / "out") == []
    assert frame_row_count() == 3

    assert main(["search", "-q", "a bicycle", "--index-dir", index]) == 0
    assert "one.mp4" in capsys.readouterr().out


def test_main_removes_deleted_video_from_search(tmp_path: Path, videos_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out, index = str(tmp_path / "out"), str(tmp_path / "index")
    write_synthetic_video(videos_dir / "two.mp4", fps=10.0, frame_count=60)
    assert main(["extract", "--input-dir", str(videos_dir), "--output-dir", out]) == 0
    assert main(["index", "--output-dir", out, "--index-dir", index]) == 0

    (videos_dir / "one.mp4").unlink()
    capsys.readouterr()
    assert main(["index", "--output-dir", out, "--index-dir", index]) == 0
    assert "Removed deleted video" in capsys.readouterr().out

    assert main(["search", "-q", "a bicycle", "--index-dir", index]) == 0
    output = capsys.readouterr().out
    assert "two.mp4" in output and "one.mp4" not in output


def test_main_indexes_second_video_incrementally(tmp_path: Path, videos_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out, index = str(tmp_path / "out"), str(tmp_path / "index")
    assert main(["extract", "--input-dir", str(videos_dir), "--output-dir", out]) == 0
    assert main(["index", "--output-dir", out, "--index-dir", index]) == 0

    write_synthetic_video(videos_dir / "two.mp4", fps=10.0, frame_count=60)
    assert main(["extract", "--video", str(videos_dir / "two.mp4"), "--output-dir", out]) == 0
    assert main(["index", "--output-dir", out, "--index-dir", index]) == 0
    capsys.readouterr()

    assert main(["search", "-q", "a bicycle", "--index-dir", index]) == 0
    output = capsys.readouterr().out
    assert "one.mp4" in output and "two.mp4" in output
