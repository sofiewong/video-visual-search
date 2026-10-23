"""Tests for grouping matching frames into clip time ranges."""

from __future__ import annotations

import pytest

from src.clip_grouping import group_matching_frames


def frame(video_id: str, t: float, duration: float = 60.0) -> dict:
    return {
        "video_id": video_id,
        "video_filename": f"{video_id}.mp4",
        "timestamp_sec": t,
        "duration_sec": duration,
    }


def test_rejects_mismatched_scores() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        group_matching_frames([frame("a", 0)], [0.5, 0.5])


def test_no_matches_returns_empty() -> None:
    records = [frame("a", 0), frame("a", 2)]
    assert group_matching_frames(records, [0.1, 0.1]) == []


def test_consecutive_frames_form_one_clip() -> None:
    records = [frame("a", 0), frame("a", 2), frame("a", 4)]
    clips = group_matching_frames(records, [0.3, 0.5, 0.4])
    assert len(clips) == 1
    clip = clips[0]
    assert clip["video_id"] == "a"
    assert clip["video_filename"] == "a.mp4"
    assert clip["start_sec"] == 0.0  # clamped at 0
    assert clip["end_sec"] == pytest.approx(5.0)
    assert clip["score"] == pytest.approx(0.4)
    assert clip["peak_score"] == pytest.approx(0.5)
    assert clip["matched_frames"] == 3


def test_single_missing_frame_does_not_split_clip() -> None:
    records = [frame("a", 0), frame("a", 2), frame("a", 4)]
    clips = group_matching_frames(records, [0.3, 0.1, 0.3])
    assert len(clips) == 1
    assert clips[0]["matched_frames"] == 2


def test_large_gap_splits_clips() -> None:
    records = [frame("a", 0), frame("a", 2), frame("a", 20), frame("a", 22)]
    clips = group_matching_frames(records, [0.3, 0.3, 0.5, 0.5])
    assert len(clips) == 2
    assert [c["start_sec"] for c in clips] == [pytest.approx(19.0), 0.0]


def test_groups_below_min_frames_are_dropped() -> None:
    records = [frame("a", 0), frame("a", 20)]
    assert group_matching_frames(records, [0.9, 0.9]) == []
    assert len(group_matching_frames(records, [0.9, 0.9], min_frames=1)) == 2


def test_end_is_clamped_to_duration() -> None:
    records = [frame("a", 8, duration=9.0), frame("a", 10, duration=9.0)]
    clips = group_matching_frames(records, [0.5, 0.5])
    assert clips[0]["end_sec"] == pytest.approx(9.0)


def test_unsorted_input_and_multiple_videos() -> None:
    records = [frame("b", 2), frame("a", 2), frame("b", 0), frame("a", 0)]
    clips = group_matching_frames(records, [0.9, 0.3, 0.8, 0.3])
    assert [c["video_id"] for c in clips] == ["b", "a"]  # sorted by score
    assert clips[0]["start_sec"] == 0.0
    assert clips[0]["matched_frames"] == 2


def test_custom_threshold() -> None:
    records = [frame("a", 0), frame("a", 2)]
    assert group_matching_frames(records, [0.5, 0.5], threshold=0.6) == []
