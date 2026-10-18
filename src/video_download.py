"""
Video downloader for testing

Dataset: https://huggingface.co/datasets/HuggingFaceFV/finevideo#how-to-download-and-use-finevideo
"""
from datasets import load_dataset
from huggingface_hub import login
from pathlib import Path
import json

login()
# Load FineVideo in streaming mode
dataset = load_dataset(
    "HuggingFaceFV/finevideo",
    split="train",
    streaming=True
)

# Retrieve one sample
sample = next(iter(dataset))

# Inspect the video's metadata
metadata = sample["json"]
print("Video metadata:")
print(json.dumps(metadata, indent=2, ensure_ascii=False))

# Determine the project root
# Go back to main folder
project_root = Path(__file__).resolve().parent.parent

# Create the data/videos directory
video_dir = project_root / "data" / "videos"
video_dir.mkdir(parents=True, exist_ok=True)

# Save video
video_path = video_dir / "sample.mp4"
video_path.write_bytes(sample["mp4"])

print(f"\nVideo saved successfully!")
print(f"Path: {video_path}")
