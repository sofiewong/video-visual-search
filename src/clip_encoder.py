"""CLIP ViT-B/32 image and text embeddings (512-d, L2-normalized)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import clip
except ImportError as exc:
    raise ImportError(
        "The OpenAI CLIP package is required. Install with: pip install openai-clip torch"
    ) from exc

CLIP_DIM = 512
DEFAULT_MODEL = "ViT-B/32"


class CLIPEncoder:
    #Encode images and text with CLIP ViT-B/32 into 512-d unit vectors.

    dim: int = CLIP_DIM

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def encode_images(
        self,
        image_paths: list[str] | list[Path],
        batch_size: int = 32,
    ) -> np.ndarray:
        #Return (N, 512) array of normalized image embeddings.
        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(image_paths), batch_size):
            #split image in batches
            batch_paths = image_paths[start : start + batch_size]
            images = []
            for path in batch_paths:
                #preprocess image and convert to tensor
                with Image.open(path) as img:
                    images.append(self.preprocess(img.convert("RGB")))

            #combine tensor
            image_tensor = torch.stack(images).to(self.device)

            #encode image and normalize
            with torch.no_grad():
                embeddings = self.model.encode_image(image_tensor)
                embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)
            #move embeddings back to numpy
            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))

        if not all_embeddings:
            return np.empty((0, CLIP_DIM), dtype=np.float32)

        #combine and return embeddings
        stacked = np.concatenate(all_embeddings, axis=0)
        if stacked.shape[1] != CLIP_DIM:
            raise RuntimeError(
                f"Expected CLIP embeddings of dim {CLIP_DIM}, got {stacked.shape[1]}"
            )
        return stacked

    def encode_text(self, query: str) -> np.ndarray:
        #Return a (1, 512) array of text embeddings for a natural-language query.
        if not query.strip():
            raise ValueError("Query text must be non-empty.")

        #encode text query
        tokens = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_text(tokens)
            embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
        #move embeddings back to numpy
        vector = embedding.cpu().numpy().astype(np.float32)
        if vector.shape != (1, CLIP_DIM):
            raise RuntimeError(
                f"Expected text embedding shape (1, {CLIP_DIM}), got {vector.shape}"
            )
        return vector
