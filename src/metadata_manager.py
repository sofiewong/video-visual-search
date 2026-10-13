"""CSV metadata for extracted video frames."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

from src.video_processor import FrameRecord

METADATA_FIELDS: Sequence[str] = (
    "frame_id",
    "video_id",
    "video_filename",
    "video_path",
    "frame_filename",
    "frame_path",
    "frame_number",
    "timestamp_sec",
    "duration_sec",
    "fps",
    "width",
    "height",
    "interval_sec",
)


def record_to_row(record: FrameRecord) -> dict[str, str | int | float]:
    """Convert a frame record to a CSV-serializable mapping."""
    return {
        "frame_id": record.frame_id,
        "video_id": record.video_id,
        "video_filename": record.video_filename,
        "video_path": record.video_path,
        "frame_filename": record.frame_filename,
        "frame_path": record.frame_path,
        "frame_number": record.frame_number,
        "timestamp_sec": record.timestamp_sec,
        "duration_sec": record.duration_sec,
        "fps": record.fps,
        "width": record.width,
        "height": record.height,
        "interval_sec": record.interval_sec,
    }


def metadata_path(output_dir: Path) -> Path:
    """Return the default metadata CSV path under ``output_dir``."""
    return output_dir / "metadata.csv"


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METADATA_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in METADATA_FIELDS})


def replace_video_records(
    csv_path: Path,
    video_id: str,
    records: Sequence[FrameRecord],
) -> None:
    """Replace any existing rows for ``video_id`` with ``records``.

    Keeps metadata consistent when the same video is processed again
    and when multiple videos are written to one CSV.
    """
    existing = [row for row in _read_rows(csv_path) if row.get("video_id") != video_id]
    new_rows = [record_to_row(record) for record in records]
    _write_rows(csv_path, existing + new_rows)


def load_records(csv_path: Path) -> list[dict[str, str]]:
    """Load all metadata rows from ``csv_path``."""
    return _read_rows(csv_path)
