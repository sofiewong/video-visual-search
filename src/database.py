
"""Build sqlite database for storing frame metadata"""

import json
import sqlite3
from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

DB_PATH = Path("data/processed/metadata.db")


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    """Create the frames table if it does not exist."""
    with get_connection() as conn:
        #table for frame metadata
        conn.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                faiss_id INTEGER PRIMARY KEY,
                frame_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                frame_path TEXT NOT NULL,
                timestamp_sec REAL NOT NULL,
                frame_number INTEGER NOT NULL,
                metadata_json TEXT NOT NULL
            )
        """)

        #table to lookup if videos has been indexed or not
        conn.execute("""
        CREATE TABLE IF NOT EXISTS indexed_videos (
            video_id TEXT PRIMARY KEY,
            indexed_at TEXT NOT NULL,
            frame_count INTEGER NOT NULL
        )
    """)

        #index for timestamp
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_video_timestamp
            ON frames (video_id, timestamp_sec)
        """)


def record_to_dict(record: Any) -> dict:
    """Convert a dataclass or dictionary to a dictionary."""
    if is_dataclass(record) and not isinstance(record, type):
        return asdict(record)

    if isinstance(record, dict):
        return record

    raise TypeError(
        f"Expected FrameRecord or dict, got {type(record).__name__}"
    )


def save_frame_metadata(records: list[Any], append: bool = False,) -> None:
    """Replace frame metadata, assigning IDs in FAISS vector order."""
    rows = []

    with get_connection() as conn:
        if append:
            row = conn.execute(
                "SELECT COALESCE(MAX(faiss_id), -1) AS max_id FROM frames"
            ).fetchone()
            starting_id = row["max_id"] + 1
        else:
            starting_id = 0
            conn.execute("DELETE FROM frames")

        for offset, record in enumerate(records):
            data = record_to_dict(record)
            faiss_id = starting_id + offset

            rows.append((
                faiss_id,
                data["frame_id"],
                data["video_id"],
                data["frame_path"],
                float(data["timestamp_sec"]),
                int(data["frame_number"]),
                json.dumps(data),
            ))

        conn.executemany("""
            INSERT INTO frames (
                faiss_id,
                frame_id,
                video_id,
                frame_path,
                timestamp_sec,
                frame_number,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, rows)



def get_all_frame_metadata() -> list[dict]:
    """Return all frame metadata in FAISS vector order."""
    with get_connection() as conn:
        results = conn.execute("""
            SELECT faiss_id, metadata_json
            FROM frames
            ORDER BY faiss_id
        """).fetchall()

    records = [json.loads(row["metadata_json"]) for row in results]

    # Ensure FAISS IDs are contiguous and aligned with embedding rows.
    faiss_ids = [row["faiss_id"] for row in results]
    if faiss_ids != list(range(len(results))):
        raise RuntimeError(
            "SQLite FAISS IDs are not contiguous from 0. "
            "Rebuild the metadata and FAISS index together."
        )

    return records

def replace_video_metadata(video_id: str, records: list[Any]) -> None:
    """Replace SQLite metadata for one video."""
    rows = []

    for record in records:
        data = record_to_dict(record)
        rows.append((
            data["frame_id"],
            data["video_id"],
            data["frame_path"],
            float(data["timestamp_sec"]),
            int(data["frame_number"]),
            json.dumps(data),
        ))

    with get_connection() as conn:
        conn.execute(
            "DELETE FROM frames WHERE video_id = ?",
            (video_id,),
        )

        conn.executemany(
            """
            INSERT INTO frames (
                frame_id, video_id, frame_path,
                timestamp_sec, frame_number, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

def load_unindexed_frame_records() -> list[dict]:
    """Load frames belonging to videos not yet indexed."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT f.faiss_id, f.video_id, f.metadata_json
            FROM frames AS f
            LEFT JOIN indexed_videos AS v
                ON f.video_id = v.video_id
            WHERE v.video_id IS NULL
            ORDER BY f.faiss_id
        """).fetchall()

    return [
        {
            "faiss_id": row["faiss_id"],
            "video_id": row["video_id"],
            "record": json.loads(row["metadata_json"]),
        }
        for row in rows
    ]


def mark_video_indexed(video_id: str, frame_count: int) -> None:
    """Mark a video as indexed after its vectors have been saved."""
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO indexed_videos
                (video_id, indexed_at, frame_count)
            VALUES (?, ?, ?)
        """, (
            video_id,
            datetime.now(timezone.utc).isoformat(),
            frame_count,
        ))