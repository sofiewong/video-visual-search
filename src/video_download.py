"""
Video downloader for testing

Dataset: https://huggingface.co/datasets/HuggingFaceFV/finevideo#how-to-download-and-use-finevideo
"""
from datasets import load_dataset
from pathlib import Path
from datetime import datetime
import secrets
import json
import random
import time

start = time.time()

# 1. Determine the project root
# Assumes this script is inside a subfolder, e.g. src/
project_root = Path(__file__).resolve().parent.parent

# 2. Create the output directory
video_dir = project_root / "data" / "videos"
video_dir.mkdir(parents=True, exist_ok=True)

# 3. Load the dataset in streaming mode
print("Connecting to FineVideo...", flush=True)

dataset = load_dataset(
    "HuggingFaceFV/finevideo",
    split="train",
    streaming=True
)

print(f"Dataset initialized in {time.time() - start:.1f}s", flush=True)

# 4. Select a random position without a shuffle buffer
# This skips a random number of samples, so it can still take time.
skip_count = random.randint(0, 100)

print(f"Selecting a sample (skip={skip_count})...", flush=True)

sample = next(iter(dataset.skip(skip_count)))

print(f"Sample retrieved in {time.time() - start:.1f}s", flush=True)

# 5. Print metadata
metadata = sample["json"]
print("Video metadata:")
print(json.dumps(metadata, indent=2, ensure_ascii=False))

# 6. Generate a unique filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
suffix = secrets.token_hex(3)

video_path = video_dir / f"sample_{timestamp}_{suffix}.mp4"

# 7. Save the video
print("Saving video...", flush=True)
video_path.write_bytes(sample["mp4"])

print(f"Done! Total time: {time.time() - start:.1f}s")
print(f"Saved to: {video_path}")

