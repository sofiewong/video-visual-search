"""Command-line entry point for extract, index, and search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.index_builder import build_index, load_unindexed_records, remove_deleted_videos
from src.search import format_hits, search_query
from src.database import (
    get_frame_paths,
    initialize_database,
    is_video_indexed,
    replace_video_metadata,
)
from src.video_processor import (
    VideoProcessingError,
    make_video_id,
    process_video,
    resolve_video_targets,
)

COMMANDS = {"extract", "index", "search"}


def _normalize_argv(argv: list[str] | None) -> list[str]:
    """Treat a missing subcommand as ``extract`` so older flags still work."""
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in COMMANDS or argv[0] in {"-h", "--help"}:
        return argv
    return ["extract", *argv]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, including extract / index / search subcommands."""
    parser = argparse.ArgumentParser(
        description="Visual Search for Videos: extract frames, index with CLIP+FAISS, search.",
    )
    subparsers = parser.add_subparsers(dest="command")

    extract = subparsers.add_parser("extract", help="Extract timestamped frames from videos.")
    extract.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/videos"),
        help="Directory containing input videos (default: data/videos).",
    )
    extract.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for frames and metadata.csv (default: data/processed).",
    )
    extract.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between sampled frames (default: 2).",
    )
    extract.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional single video file instead of processing --input-dir.",
    )

    index = subparsers.add_parser("index", help="Encode frames with CLIP and build a FAISS index.")
    index.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Processed-data directory with metadata.csv and frames/.",
    )
    index.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/index"),
        help="Where to write frames.faiss (default: data/index).",
    )
    index.add_argument("--batch-size", type=int, default=32)

    search = subparsers.add_parser("search", help="Search frames with a natural-language query.")
    search.add_argument(
        "--query",
        "-q",
        default=None,
        help='Search text, e.g. "a person riding a bicycle". Prompts if omitted.',
    )
    search.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/index"),
        help="Directory containing frames.faiss (default: data/index).",
    )
    search.add_argument("--top-k", type=int, default=5, help="Number of videos (default: 5).")

    args = parser.parse_args(_normalize_argv(argv))
    if args.command is None:
        parser.print_help()
        raise SystemExit(0)
    return args


def validate_extract_args(args: argparse.Namespace) -> None:
    """Raise SystemExit with a useful message for invalid extract arguments."""
    if args.interval <= 0:
        raise SystemExit("--interval must be a positive number of seconds.")
    if args.video is not None and not args.video.exists():
        raise SystemExit(f"Video file does not exist: {args.video}")
    if args.video is None and not args.input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    if args.video is None and args.input_dir.exists() and not args.input_dir.is_dir():
        raise SystemExit(f"--input-dir is not a directory: {args.input_dir}")


def run_extract(args: argparse.Namespace) -> int:
    """Process videos and write frames plus metadata."""
    validate_extract_args(args)
    output_dir = args.output_dir.resolve()
    frames_dir = output_dir / "frames"
    DB_PATH =  output_dir / "metadata.db"
    frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        targets, skipped = resolve_video_targets(
            args.input_dir.resolve() if args.video is None else args.input_dir,
            args.video.resolve() if args.video is not None else None,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not targets:
        print("No supported videos found (.mp4, .avi, .mov).", file=sys.stderr)
        for message in skipped:
            print(f"  {message}", file=sys.stderr)
        return 1

    
    succeeded = 0
    failed = 0
    already_indexed = 0
    already_extracted = 0
    frames_written = 0
    failures: list[str] = []

    initialize_database()

    for index, path in enumerate(targets, start=1):
        print(f"Processing {path.name} ({index}/{len(targets)})...")
        video_id = make_video_id(path)
        #skip indexed videos so their FAISS rows stay aligned
        if is_video_indexed(video_id):
            already_indexed += 1
            print("  Skipped: already indexed")
            continue
        #skip extracted videos unless frame images are missing
        frame_paths = get_frame_paths(video_id)
        if frame_paths and all(Path(p).is_file() for p in frame_paths):
            already_extracted += 1
            print("  Skipped: already extracted")
            continue
        try:
            video, records = process_video(path, frames_dir, args.interval)
            replace_video_metadata(video.video_id, records)
        except VideoProcessingError as exc:
            failed += 1
            failures.append(str(exc))
            print(f"  Failed: {exc}")
            continue

        succeeded += 1
        frames_written += len(records)
        print(
            f"  OK: {len(records)} frames, duration={video.duration_sec:.2f}s, "
            f"fps={video.fps:.2f}, {video.width}x{video.height}"
        )

    print()
    print("Summary")
    print(f"  Videos processed: {succeeded}")
    print(f"  Already extracted: {already_extracted}")
    print(f"  Already indexed:  {already_indexed}")
    print(f"  Videos failed:    {failed}")
    print(f"  Frames extracted: {frames_written}")
    print(f"  Metadata database: {DB_PATH.resolve()}")
    if skipped:
        print("  Skipped files:")
        for message in skipped:
            print(f"    {message}")
    if failures:
        print("  Failures:")
        for message in failures:
            print(f"    {message}")

    return 0 if succeeded or already_indexed or already_extracted else 1


def run_index(args: argparse.Namespace) -> int:
    """Encode extracted frames and write a FAISS index."""
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be a positive integer.")
    try:
        #drop embeddings and metadata of videos removed from disk
        for video_path in remove_deleted_videos(args.index_dir):
            print(f"Removed deleted video: {video_path}")

        records = load_unindexed_records()

        if not records:
            print("All videos are already indexed.")
            return 0

        count = build_index(
            records=records,
            index_dir=args.index_dir,
            frames_dir=args.output_dir / "frames",
            batch_size=args.batch_size,
        )
        index_path = Path(args.index_dir) / "frames.faiss"
        print(f"Indexed frames: {count}")
        print(f"FAISS index: {index_path.resolve()}")
        print(f"Index file exists: {index_path.is_file()}")

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def run_search(args: argparse.Namespace) -> int:
    """Run a CLIP text query against the FAISS index."""
    if args.top_k <= 0:
        raise SystemExit("--top-k must be a positive integer.")
    query = args.query if args.query is not None else input("Enter a search query: ")
    try:
        hits = search_query(query, index_dir=args.index_dir, top_k=args.top_k)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(format_hits(query.strip(), hits))
    return 0


def run(args: argparse.Namespace) -> int:
    """Dispatch to extract, index, or search."""
    if args.command == "extract":
        return run_extract(args)
    if args.command == "index":
        return run_index(args)
    if args.command == "search":
        return run_search(args)
    raise SystemExit(f"Unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point used by ``python -m src.main``."""
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
