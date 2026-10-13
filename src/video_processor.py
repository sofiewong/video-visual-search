"""OpenCV-based video validation and frame extraction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov"}


class VideoProcessingError(Exception):
    """Raised when a video file cannot be opened or processed."""


@dataclass(frozen=True)
class VideoInfo:
    """Properties of a readable video file."""

    path: Path
    video_id: str
    filename: str
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int


@dataclass(frozen=True)
class FrameRecord:
    """Metatdata of extracted frame and source-video metadata"""

    frame_id: str
    video_id: str
    video_filename: str
    video_path: str
    frame_filename: str
    frame_path: str
    frame_number: int
    timestamp_sec: float
    duration_sec: float
    fps: float
    width: int
    height: int
    interval_sec: float


def make_video_id(path: Path) -> str:
    #Return a stable ID from the filename stem and a short path hash.
    resolved = path.resolve()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", resolved.stem).strip("._-") or "video"
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}"


def list_videos(input_dir: Path) -> list[Path]:
    #List supported video files in "input_dir"
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    return [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def frame_number_to_timestamp(frame_number: int, fps: float) -> float:
    #Convert frame index to timestamp
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if frame_number < 0:
        raise ValueError(f"frame_number must be >= 0, got {frame_number}")
    return frame_number / fps


def sample_frame_indices(fps: float, frame_count: int, interval_sec: float) -> list[int]:
    #Return frame indices sampled every "interval_sec" seconds.
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if interval_sec <= 0:
        raise ValueError(f"interval_sec must be positive, got {interval_sec}")

    indices: list[int] = []
    timestamp = 0.0
    duration = (frame_count - 1) / fps
    while timestamp <= duration + 1e-9:
        #append frame index (timestamp*fps)
        frame_number = int(round(timestamp * fps))
        frame_number = min(max(frame_number, 0), frame_count - 1)
        #avoid duplicate
        if not indices or indices[-1] != frame_number:
            indices.append(frame_number)
        timestamp += interval_sec
    return indices


def validate_video(path: Path) -> VideoInfo:
    #Open video in path with OpenCV and return video properties
    if not path.is_file():
        raise VideoProcessingError(f"Video file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise VideoProcessingError(
            f"Unsupported video format '{path.suffix}' for {path.name}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    #open video
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoProcessingError(
                f"Unable to open video (unreadable or corrupted): {path}"
            )

        #get and convert video properties
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise VideoProcessingError(
                f"Video has invalid properties (fps={fps}, frames={frame_count}, "
                f"size={width}x{height}): {path}"
            )

        #calculate video duration
        duration_sec = frame_count / fps
        return VideoInfo(
            path=path.resolve(),
            video_id=make_video_id(path),
            filename=path.name,
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration_sec,
            width=width,
            height=height,
        )
    finally:
        capture.release()


def _write_frame(image: np.ndarray, destination: Path) -> None:
    #write image of frame to disk
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise VideoProcessingError(f"Failed to write frame image: {destination}")


def extract_frames(
    video: VideoInfo,
    frames_dir: Path,
    interval_sec: float,
) -> list[FrameRecord]:
    #Extract sampled frames sequentially and return metadata records.

    #assign indices for sampled frames
    indices = sample_frame_indices(video.fps, video.frame_count, interval_sec)
    #use set for faster lookup
    wanted = set(indices)
    records: list[FrameRecord] = []

    #open video
    capture = cv2.VideoCapture(str(video.path))
    if not capture.isOpened():
        raise VideoProcessingError(
            f"Unable to open video (unreadable or corrupted): {video.path}"
        )

    try:
        frame_idx = 0
        while True:
            #read frame
            ok, image = capture.read()
            if not ok:
                break
            if frame_idx in wanted:
                #store metadata if frame is sampled
                timestamp = frame_number_to_timestamp(frame_idx, video.fps)
                #filename = id + _ + frame no. padded w/ 8 zeros
                frame_filename = f"{video.video_id}_f{frame_idx:08d}.jpg"
                #construct path
                frame_path = (frames_dir / frame_filename).resolve()
                #save image
                _write_frame(image, frame_path)
                records.append(
                    FrameRecord(
                        frame_id=f"{video.video_id}_{frame_idx}",
                        video_id=video.video_id,
                        video_filename=video.filename,
                        video_path=str(video.path),
                        frame_filename=frame_filename,
                        frame_path=str(frame_path),
                        frame_number=frame_idx,
                        timestamp_sec=round(timestamp, 6),
                        duration_sec=round(video.duration_sec, 6),
                        fps=video.fps,
                        width=video.width,
                        height=video.height,
                        interval_sec=interval_sec,
                    )
                )
                #remove extracted frame from set
                wanted.remove(frame_idx)
                if not wanted:
                    #break when all frames are extracted
                    break
            frame_idx += 1
    finally:
        capture.release()

    if not records:
        raise VideoProcessingError(
            f"No frames could be extracted from video: {video.path}"
        )
    return records


def process_video(
    path: Path,
    frames_dir: Path,
    interval_sec: float,
) -> tuple[VideoInfo, list[FrameRecord]]:
    #Validate a video and extract sampled frames
    video = validate_video(path)
    records = extract_frames(video, frames_dir, interval_sec)
    return video, records


def resolve_video_targets(
    input_dir: Path,
    video: Path | None = None,
) -> tuple[list[Path], list[str]]:
    #Return videos to process and skipped-file messages
    skipped: list[str] = []
    if video is not None:
        if not video.is_file():
            raise FileNotFoundError(f"Video file does not exist: {video}")
        return [video], skipped

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    videos = list_videos(input_dir)
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped.append(
                f"{path.name}: skipped unsupported format '{path.suffix}'"
            )
    return videos, skipped
