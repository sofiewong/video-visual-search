
from collections import defaultdict


def group_matching_frames(
    records: list[dict],
    scores,
    threshold: float = 0.20,
    sample_interval_sec: float = 2.0,
    merge_gap_sec: float = 2.0,
    min_frames: int = 2,
) -> list[dict]:
    """
    Group matching frames into time ranges within each video.

    records: metadata for every frame
    scores: CLIP text-image similarity scores in the same order
    threshold: minimum similarity score for a frame to match
    sample_interval_sec: interval between sampled frames
    merge_gap_sec: maximum gap between ranges that will be merged
    min_frames: minimum number of matching frames in a result
    """
    if len(records) != len(scores):
        raise ValueError("Each frame must have exactly one similarity score.")

    # Group frames by their source video.
    videos = defaultdict(list)

    for record, score in zip(records, scores):
        videos[record["video_id"]].append(
            (record, float(score))
        )

    clips = []

    for video_id, frames in videos.items():
        # Process frames in chronological order.
        frames.sort(key=lambda item: float(item[0]["timestamp_sec"]))

        #find matched frames
        matching = [
            (record, score)
            for record, score in frames
            if score >= threshold
        ]

        if not matching:
            continue

        groups = []
        current_group = [matching[0]]

        for item in matching[1:]:
            record, score = item
            previous_record, _ = current_group[-1]

            time_gap = (
                float(record["timestamp_sec"])
                - float(previous_record["timestamp_sec"])
            )

            if 0 <= time_gap <= sample_interval_sec + merge_gap_sec:
                current_group.append(item)
            else:
                groups.append(current_group)
                current_group = [item]

        groups.append(current_group)

        # Convert groups of frames into clip ranges.
        for group in groups:
            if len(group) < min_frames:
                continue

            first_record = group[0][0]
            last_record = group[-1][0]

            duration = float(first_record["duration_sec"])

            start_sec = max(
                0.0,
                float(first_record["timestamp_sec"])
                - sample_interval_sec / 2,
            )

            end_sec = min(
                duration,
                float(last_record["timestamp_sec"])
                + sample_interval_sec / 2,
            )

            group_scores = [score for _, score in group]

            clips.append({
                "video_id": video_id,
                "video_filename": first_record["video_filename"],
                "start_sec": start_sec,
                "end_sec": end_sec,
                "score": sum(group_scores) / len(group_scores),
                "peak_score": max(group_scores),
                "matched_frames": len(group),
            })

    # Show the most relevant clips first.
    clips.sort(key=lambda clip: clip["score"], reverse=True)

    return clips
