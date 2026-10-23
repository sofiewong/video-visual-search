"""Build a FAISS index of CLIP frame embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from src.database import (
    delete_videos,
    get_frame_video_ids,
    get_video_paths,
    load_unindexed_frame_records,
    mark_video_indexed,
    save_frame_metadata,
)
from src.video_processor import FrameRecord


INDEX_FILENAME = "frames.faiss"
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


def save_index(
    index_dir: Path,
    embeddings: np.ndarray,
    records: list[FrameRecord],
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

    index_path = index_dir / INDEX_FILENAME
    embeddings_path = index_dir / EMBEDDING_FILENAME
    config_path = index_dir / CONFIG_FILENAME
    #create index that search using cosine similarity (inner product of L2 norm vectors)
    if index_path.is_file():
        index = faiss.read_index(str(index_path))

        if index.d != embeddings.shape[1]:
            raise ValueError("Embedding dimension does not match existing index.")
        starting_id = index.ntotal

        if not embeddings_path.is_file():
            raise FileNotFoundError(
                f"Existing embeddings file not found: {embeddings_path}"
            )

        old_embeddings = np.load(embeddings_path)
        if old_embeddings.shape[0] != starting_id:
            raise ValueError(
                "Existing embedding count does not match FAISS index size."
            )
        
        # Append new vectors to the existing index.
        index.add(embeddings)

        # Preserve old embeddings and append new ones.
        all_embeddings = np.concatenate(
            [old_embeddings, embeddings],
            axis=0,
        )

    else:
        # First index build.
        starting_id = 0
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        all_embeddings = embeddings

    # Save metadata of frames aligned with their FAISS IDs.
    save_frame_metadata(records, starting_id=starting_id)

    # Save updated index and embeddings.
    faiss.write_index(index, str(index_path))
    np.save(embeddings_path, all_embeddings)

    # Mark each video as indexed after saving the embeddings.
    video_counts = {}

    for record in records:
        video_counts[record.video_id] = (
            video_counts.get(record.video_id, 0) + 1
        )

    for video_id, frame_count in video_counts.items():
        mark_video_indexed(video_id, frame_count)

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
    records: list[FrameRecord],
    index_dir: str | Path = "data/index",
    frames_dir: str | Path | None = None,
    batch_size: int = 32,
    encoder: ImageEncoder | None = None,
    delete_frames: bool = True,
) -> int:
    """
    Encode each frame to a 512-d CLIP vector and store them in FAISS.
    metadata_csv: path of metadata record
    index_dir: directory to save the index to
    frames_dir: optional frame directory
    batch_size: size of batch to process
    delete_frames: delete frame images after their embeddings are saved
    Returns the number of indexed frames.
    """
    
    index_dir = Path(index_dir)
    frames_dir_path = (
        Path(frames_dir)
        if frames_dir is not None
        else None
    )

    #load metadata of frames
    valid_records = []
    paths = []
    skipped = []
    for record in records:
        path = Path(record.frame_path)

        # First try the path as stored in the record.
        if not path.is_file() and frames_dir_path is not None:
            path = frames_dir_path / path.name

        if not path.is_file():
            skipped.append(f"Missing frame: {record.frame_path}")
            continue

        valid_records.append(record)
        paths.append(path)

    for message in skipped:
        print(f"  Skipping {message}")

    if not valid_records:
        raise ValueError(
            "No valid frame images found. Run frame extraction first."
        )

    #encode frames
    encoder = encoder or _default_encoder()
    print(f"Encoding {len(paths)} frames with CLIP (batch_size={batch_size})...")
    embeddings = encoder.encode_images([str(path) for path in paths], batch_size=batch_size)
    if embeddings.shape[0] != len(valid_records):
        raise RuntimeError("Encoder returned a different number of vectors than frames.")

    #get model name
    model_name = getattr(encoder, "model_name", "ViT-B/32")
    dim = int(getattr(encoder, "dim", embeddings.shape[1]))
    if embeddings.shape[1] != dim:
        raise RuntimeError(f"Expected embedding dim {dim}, got {embeddings.shape[1]}")

    #save to index directory
    save_index(index_dir, embeddings, valid_records, model_name=str(model_name))
    print(f"Indexed {len(valid_records)} frames ({embeddings.shape[1]}-d).")
    print(f"Saved FAISS index to {index_dir / INDEX_FILENAME}")

    #delete frame images now that embeddings are saved
    if delete_frames:
        deleted = 0
        for path in paths:
            try:
                path.unlink()
                deleted += 1
            except OSError as exc:
                print(f"  Could not delete {path}: {exc}")
        print(f"Deleted {deleted} frame images.")

    return len(valid_records)


def remove_deleted_videos(index_dir: str | Path = "data/index") -> list[str]:
    """
    Delete embeddings and metadata of videos whose source files no longer exist.
    index_dir: directory containing the FAISS index
    Returns the paths of the removed videos.
    """
    removed = {
        video_id: video_path
        for video_id, video_path in get_video_paths().items()
        if video_path and not Path(video_path).is_file()
    }
    if not removed:
        return []

    index_dir = Path(index_dir)
    index_path = index_dir / INDEX_FILENAME
    embeddings_path = index_dir / EMBEDDING_FILENAME
    if index_path.is_file():
        index = faiss.read_index(str(index_path))
        embeddings = np.load(embeddings_path)
        if embeddings.shape[0] != index.ntotal:
            raise ValueError(
                "Existing embedding count does not match FAISS index size."
            )

        #keep vectors of frames whose video still exists
        indexed_video_ids = get_frame_video_ids()[: index.ntotal]
        keep = np.array(
            [video_id not in removed for video_id in indexed_video_ids],
            dtype=bool,
        )
        kept_embeddings = embeddings[keep]

        #rebuild the index from the remaining vectors
        new_index = faiss.IndexFlatIP(index.d)
        new_index.add(kept_embeddings)
        faiss.write_index(new_index, str(index_path))
        np.save(embeddings_path, kept_embeddings)

    #delete metadata and any leftover frame images
    for frame_path in delete_videos(set(removed)):
        Path(frame_path).unlink(missing_ok=True)

    return sorted(removed.values())


def load_unindexed_records() -> list[FrameRecord]:
    """Return frame records for extracted videos that are not indexed yet."""
    return [
        FrameRecord(**{
            key: value
            for key, value in item["record"].items()
            if key in FrameRecord.__dataclass_fields__
        })
        for item in load_unindexed_frame_records()
    ]


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Build a FAISS index from extracted frames.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Processed-data directory containing frames/.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/index"),
        help="Directory to write frames.faiss and sidecar metadata.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        for video_path in remove_deleted_videos(args.index_dir):
            print(f"Removed deleted video: {video_path}")

        records = load_unindexed_records()
        if not records:
            print("All videos are already indexed.")
            return 0

        build_index(
            records=records,
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
