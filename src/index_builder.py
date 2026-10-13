"""Build a FAISS index of CLIP frame embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np

from src.metadata_manager import load_records, metadata_path

INDEX_FILENAME = "frames.faiss"
METADATA_FILENAME = "frame_metadata.json"
CONFIG_FILENAME = "index_config.json"
EMBEDDING_FILENAME = "frame_embeddings.npy"


class ImageEncoder(Protocol):
    """Minimal encoder interface used by the index builder."""

    dim: int

    def encode_images(
        self,
        image_paths: list[str] | list[Path],
        batch_size: int = 32,
    ) -> np.ndarray: ...


def _default_encoder() -> ImageEncoder:
    from src.clip_encoder import CLIPEncoder

    return CLIPEncoder()


def _resolve_frame_path(row: dict[str, str], frames_dir: Path | None) -> Path | None:
    #return path of image file
    raw = (row.get("frame_path") or "").strip()
    if raw:
        path = Path(raw)
        if path.is_file():
            return path
        if frames_dir is not None:
            fallback = frames_dir / path.name
            if fallback.is_file():
                return fallback
    filename = (row.get("frame_filename") or "").strip()
    if frames_dir is not None and filename:
        fallback = frames_dir / filename
        if fallback.is_file():
            return fallback
    return None


def load_indexable_rows(
    metadata_csv: Path,
    frames_dir: Path | None = None,
) -> tuple[list[dict[str, str]], list[Path], list[str]]:
    """
    Return (rows, image paths, skipped messages) for frames that exist on disk.
    metadata_csv: path of metadata record
    frames_dir: optional frame directory
    """
    if not metadata_csv.is_file():
        raise FileNotFoundError(
            f"Metadata CSV not found: {metadata_csv}. Extract frames first."
        )
    #load all rows from metadata record
    rows = load_records(metadata_csv)
    #store frames w/ image files
    valid_rows: list[dict[str, str]] = []
    #store path of image
    paths: list[Path] = []
    #stored skipped messages
    skipped: list[str] = []
    for row in rows:
        #get path
        path = _resolve_frame_path(row, frames_dir)
        if path is None:
            #record skipped message
            label = row.get("frame_filename") or row.get("frame_id") or "unknown"
            skipped.append(f"{label}: image file not found")
            continue
        #add path, frames to valid_rows
        row = dict(row)
        row["frame_path"] = str(path.resolve())
        valid_rows.append(row)
        #save image path
        paths.append(path.resolve())
    return valid_rows, paths, skipped


def save_index(
    index_dir: Path,
    embeddings: np.ndarray,
    records: list[dict[str, Any]],
    model_name: str = "ViT-B/32",
) -> faiss.Index:
    """
    Write FAISS index, aligned metadata, and embedding matrix.
    index_dir: directory to save the index to
    embeddings: image embeddings
    records: metadata of each frame
    model name: CLIP model
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2-D, got shape {embeddings.shape}")
    if embeddings.shape[0] != len(records):
        raise ValueError(
            f"Embedding count {embeddings.shape[0]} != metadata rows {len(records)}"
        )
    #convert to float
    if embeddings.dtype != np.float32:  
        embeddings = embeddings.astype(np.float32)

    #make output directory
    index_dir.mkdir(parents=True, exist_ok=True)
    #create index that search using cosine similarity (inner product of L2 norm vectors)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    #write index to directory
    faiss.write_index(index, str(index_dir / INDEX_FILENAME))
    #save embeddings to numpy
    np.save(index_dir / EMBEDDING_FILENAME, embeddings)
    #save metadata as JSON
    (index_dir / METADATA_FILENAME).write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
    config = {
        "model_name": model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "num_vectors": int(embeddings.shape[0]),
        "metric": "inner_product",
        "normalized": True,
    }
    (index_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    return index


def build_index(
    metadata_csv: str | Path = "data/processed/metadata.csv",
    index_dir: str | Path = "data/index",
    frames_dir: str | Path | None = None,
    batch_size: int = 32,
    encoder: ImageEncoder | None = None,
) -> int:
    """
    Encode each frame to a 512-d CLIP vector and store them in FAISS.
    metadata_csv: path of metadata record
    index_dir: directory to save the index to
    frames_dir: optional frame directory
    batch_size: size of batch to process
    Returns the number of indexed frames.
    """
    metadata_csv = Path(metadata_csv)
    index_dir = Path(index_dir)
    frames_dir_path = Path(frames_dir) if frames_dir is not None else metadata_csv.parent / "frames"

    #load metadata of frames
    rows, paths, skipped = load_indexable_rows(metadata_csv, frames_dir_path)
    for message in skipped:
        print(f"  Skipping {message}")
    if not rows:
        raise ValueError("No valid frame images found. Extract frames before indexing.")

    #encode frames
    encoder = encoder or _default_encoder()
    print(f"Encoding {len(paths)} frames with CLIP (batch_size={batch_size})...")
    embeddings = encoder.encode_images([str(path) for path in paths], batch_size=batch_size)
    if embeddings.shape[0] != len(rows):
        raise RuntimeError("Encoder returned a different number of vectors than frames.")

    #get model name
    model_name = getattr(encoder, "model_name", "ViT-B/32")
    dim = int(getattr(encoder, "dim", embeddings.shape[1]))
    if embeddings.shape[1] != dim:
        raise RuntimeError(f"Expected embedding dim {dim}, got {embeddings.shape[1]}")

    #save to index directory
    save_index(index_dir, embeddings, rows, model_name=str(model_name))
    print(f"Indexed {len(rows)} frames ({embeddings.shape[1]}-d).")
    print(f"Saved FAISS index to {index_dir / INDEX_FILENAME}")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Build a FAISS index from extracted frames.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to metadata.csv (default: <output-dir>/metadata.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Processed-data directory containing metadata.csv and frames/.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/index"),
        help="Directory to write frames.faiss and sidecar metadata.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    csv_path = args.metadata if args.metadata is not None else metadata_path(args.output_dir)
    try:
        build_index(
            metadata_csv=csv_path,
            index_dir=args.index_dir,
            frames_dir=args.output_dir / "frames",
            batch_size=args.batch_size,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
