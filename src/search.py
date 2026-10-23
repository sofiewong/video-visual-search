"""Natural-language search over indexed video frames."""

from __future__ import annotations
from src.clip_grouping import group_matching_frames

import json
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np

from src.index_builder import INDEX_FILENAME
from src.database import get_all_frame_metadata


class TextEncoder(Protocol):
    def encode_text(self, query: str) -> np.ndarray: ...


def _default_encoder() -> TextEncoder:
    from src.clip_encoder import CLIPEncoder

    return CLIPEncoder()


def load_faiss_index(index_dir: Path) -> tuple[faiss.Index, list[dict[str, Any]]]:
    """
    Load the FAISS index and the metadata list aligned by vector id.
    Return FAISS index and list of metadata dictionaries
    """
    index_path = index_dir / INDEX_FILENAME
    #meta_path = index_dir / METADATA_FILENAME
    if not index_path.is_file():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}. Run: python -m src.main index"
        )
    # if not meta_path.is_file():
    #     raise FileNotFoundError(f"Index metadata not found: {meta_path}")
    #get faiss index matched and metadata

    index = faiss.read_index(str(index_path))
    #records = json.loads(meta_path.read_text(encoding="utf-8"))
    records = get_all_frame_metadata()
    if index.ntotal != len(records):
        raise RuntimeError(
            f"Index/metadata mismatch: {index.ntotal} vectors vs {len(records)} rows"
        )
    return index, records


def search_query(
    query: str,
    index_dir: str | Path = "data/index",
    top_k: int = 5,
    encoder: TextEncoder | None = None,
) -> list[dict]:
    """
    Encode query with CLIP and return the top matching videos.

    Each result holds every matched clip in that video, and videos are
    ranked by their best clip score.
    """
    if not query or not query.strip():
        raise ValueError("Query text must be non-empty.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    #load FAISS index and metadata from index_dir
    index_dir = Path(index_dir)
    index, records = load_faiss_index(index_dir)
    encoder = encoder or _default_encoder()
    #convert query to CLIP embedding
    text_vector = encoder.encode_text(query.strip())
    #convert to 2D array if needed
    if text_vector.ndim == 1:
        text_vector = text_vector.reshape(1, -1)
    #convert to numpy array
    text_vector = np.asarray(text_vector, dtype=np.float32)
    if text_vector.shape[1] != index.d:
        raise RuntimeError(
            f"Query dim {text_vector.shape[1]} does not match index dim {index.d}"
        )

    # Load embeddings for every indexed frame.
    embeddings_path = index_dir / "frame_embeddings.npy"
    if not embeddings_path.is_file():
        raise FileNotFoundError(
            f"Missing {embeddings_path}. "
            "Save the frame embeddings when building the index."
        )
    #load embeddings to numpy
    embeddings = np.load(embeddings_path)
    if embeddings.shape[0] != len(records):
        raise RuntimeError(
            "The number of frame embeddings does not match metadata."
        )

    if embeddings.shape[1] != index.d:
        raise RuntimeError(
            "The saved embedding dimensions do not match the FAISS index."
        )
    # Compute cosine similarity against every frame (inner product since embeddings are L2 normalized)
    scores = embeddings @ text_vector[0]

     # Group high-scoring frames into time ranges.
    clips = group_matching_frames(
        records=records,
        scores=scores,
        threshold=0.25,
        sample_interval_sec=2.0,
        merge_gap_sec=2.0,
        min_frames=2,
    )

    return group_clips_by_video(clips)[:top_k]


def group_clips_by_video(clips: list[dict]) -> list[dict]:
    """Group clips by video, ranked by each video's best clip score."""
    videos: dict[Any, dict] = {}
    for clip in clips:
        video = videos.setdefault(clip["video_id"], {
            "video_id": clip["video_id"],
            "video_filename": clip["video_filename"],
            "video_path": clip.get("video_path"),
            "score": clip["score"],
            "clips": [],
        })
        video["score"] = max(video["score"], clip["score"])
        video["clips"].append(clip)

    #show clips within a video in chronological order
    for video in videos.values():
        video["clips"].sort(key=lambda clip: clip["start_sec"])

    return sorted(videos.values(), key=lambda video: video["score"], reverse=True)


def format_hits(query: str, hits: list[dict]) -> str:
    """print search results for the CLI"""
    if not hits:
        return f'No matches for "{query}".'
    lines = [f'Top {len(hits)} videos for "{query}":', ""]
    for rank, video in enumerate(hits, start=1):
        clips = video["clips"]
        lines.append(
            f"{rank}. {video['video_filename']}  "
            f"score={video['score']:.4f}  clips={len(clips)}"
        )

        if video.get("video_path"):
            lines.append(f"   Video: {video['video_path']}")

        for clip in clips:
            start = clip["start_sec"]
            end = clip["end_sec"]
            lines.append(
                f"   - clip={format_timestamp(start)}–{format_timestamp(end)}  "
                f"score={clip['score']:.4f}  "
                f"duration={end - start:.2f}s  "
                f"matching frames={clip['matched_frames']}"
            )

    return "\n".join(lines)

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Search indexed frames with a natural-language query.",
    )
    parser.add_argument(
        "--query",
        "-q",
        default=None,
        help='Search text, e.g. "a person riding a bicycle".',
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/index"),
        help="Directory containing frames.faiss (default: data/index).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of matching videos to return (default: 5).",
    )
    args = parser.parse_args(argv)
    query = args.query if args.query is not None else input("Enter a search query: ")
    try:
        hits = search_query(query, index_dir=args.index_dir, top_k=args.top_k)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(format_hits(query.strip(), hits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
