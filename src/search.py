"""Natural-language search over indexed video frames."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np

from src.index_builder import INDEX_FILENAME, METADATA_FILENAME


class TextEncoder(Protocol):
    def encode_text(self, query: str) -> np.ndarray: ...


@dataclass(frozen=True)
class SearchHit:
    #One FAISS match joined back to source-frame metadata.

    rank: int
    video_filename: str
    timestamp_sec: float
    similarity: float
    frame_id: str
    frame_path: str
    video_id: str
    frame_number: int


def _default_encoder() -> TextEncoder:
    from src.clip_encoder import CLIPEncoder

    return CLIPEncoder()


def load_faiss_index(index_dir: Path) -> tuple[faiss.Index, list[dict[str, Any]]]:
    #Load the FAISS index and the metadata list aligned by vector id.
    #return FAISS index and list of metadata dictionaries
    index_path = index_dir / INDEX_FILENAME
    meta_path = index_dir / METADATA_FILENAME
    if not index_path.is_file():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}. Run: python -m src.main index"
        )
    if not meta_path.is_file():
        raise FileNotFoundError(f"Index metadata not found: {meta_path}")
    #get faiss index matched and metadata
    index = faiss.read_index(str(index_path))
    records = json.loads(meta_path.read_text(encoding="utf-8"))
    if index.ntotal != len(records):
        raise RuntimeError(
            f"Index/metadata mismatch: {index.ntotal} vectors vs {len(records)} rows"
        )
    return index, records


def _as_float(value: Any, default: float = 0.0) -> float:
    #convert to float
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    #convert to int
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def record_to_hit(rank: int, score: float, record: dict[str, Any]) -> SearchHit:
    #Map a metadata row and similarity score to a search hit.
    return SearchHit(
        rank=rank,
        video_filename=str(record.get("video_filename") or ""),
        timestamp_sec=_as_float(record.get("timestamp_sec")),
        similarity=float(score),
        frame_id=str(record.get("frame_id") or ""),
        frame_path=str(record.get("frame_path") or ""),
        video_id=str(record.get("video_id") or ""),
        frame_number=_as_int(record.get("frame_number")),
    )


def search_query(
    query: str,
    index_dir: str | Path = "data/index",
    top_k: int = 5,
    encoder: TextEncoder | None = None,
) -> list[SearchHit]:
    #Encode query with CLIP and return the top matching frames.
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

    #find score and vector id of top k hits
    k = min(top_k, index.ntotal)
    scores, ids = index.search(text_vector, k)
    #store top k hits in list
    hits: list[SearchHit] = []
    for rank, (score, faiss_id) in enumerate(zip(scores[0], ids[0], strict=True), start=1):
        if faiss_id < 0:
            continue
        hits.append(record_to_hit(rank, float(score), records[int(faiss_id)]))
    return hits


def format_hits(query: str, hits: list[SearchHit]) -> str:
    #print search results for the CLI
    if not hits:
        return f'No matches for "{query}".'
    lines = [f'Top {len(hits)} matches for "{query}":', ""]
    for hit in hits:
        lines.append(
            f"{hit.rank}. {hit.video_filename}  "
            f"t={hit.timestamp_sec:.2f}s  "
            f"score={hit.similarity:.4f}"
        )
        if hit.frame_path:
            lines.append(f"   {hit.frame_path}")
    return "\n".join(lines)


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
        help="Number of matching frames to return (default: 5).",
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
