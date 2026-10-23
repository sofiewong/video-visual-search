"""Tests for CLIPEncoder using a stubbed CLIP model (no weight download)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import src.clip_encoder as clip_encoder
from src.clip_encoder import CLIP_DIM, CLIPEncoder


class FakeModel:
    def __init__(self, dim: int = CLIP_DIM) -> None:
        self.dim = dim
        self.image_batches: list[int] = []
        self.eval_called = False

    def eval(self) -> "FakeModel":
        self.eval_called = True
        return self

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        self.image_batches.append(images.shape[0])
        return torch.full((images.shape[0], self.dim), 3.0)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.full((tokens.shape[0], self.dim), 2.0)


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> FakeModel:
    model = FakeModel()
    preprocess = lambda img: torch.zeros(3, 4, 4)
    monkeypatch.setattr(clip_encoder.clip, "load", lambda name, device: (model, preprocess))
    return model


def make_images(tmp_path: Path, count: int) -> list[Path]:
    paths = []
    for i in range(count):
        path = tmp_path / f"img{i}.png"
        Image.new("RGB", (8, 8), color=(i * 10, 0, 0)).save(path)
        paths.append(path)
    return paths


def test_init_loads_model_in_eval_mode(fake_model: FakeModel) -> None:
    encoder = CLIPEncoder()
    assert encoder.model is fake_model
    assert fake_model.eval_called
    assert encoder.model_name == "ViT-B/32"
    assert encoder.dim == CLIP_DIM
    assert encoder.device in {"cpu", "cuda"}


def test_encode_images_returns_normalized_batched_vectors(fake_model: FakeModel, tmp_path: Path) -> None:
    paths = make_images(tmp_path, 5)
    vectors = CLIPEncoder().encode_images(paths, batch_size=2)
    assert vectors.shape == (5, CLIP_DIM)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)
    assert fake_model.image_batches == [2, 2, 1]


def test_encode_images_empty_list(fake_model: FakeModel) -> None:
    vectors = CLIPEncoder().encode_images([])
    assert vectors.shape == (0, CLIP_DIM)


def test_encode_images_rejects_wrong_dim(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = FakeModel(dim=256)
    monkeypatch.setattr(clip_encoder.clip, "load", lambda name, device: (model, lambda img: torch.zeros(3, 4, 4)))
    with pytest.raises(RuntimeError, match="dim"):
        CLIPEncoder().encode_images(make_images(tmp_path, 1))


def test_encode_text_returns_normalized_vector(fake_model: FakeModel) -> None:
    vector = CLIPEncoder().encode_text("a person riding a bicycle")
    assert vector.shape == (1, CLIP_DIM)
    assert vector.dtype == np.float32
    assert np.linalg.norm(vector) == pytest.approx(1.0, rel=1e-5)


def test_encode_text_rejects_empty_query(fake_model: FakeModel) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        CLIPEncoder().encode_text("   ")


def test_encode_text_rejects_wrong_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel(dim=256)
    monkeypatch.setattr(clip_encoder.clip, "load", lambda name, device: (model, lambda img: img))
    with pytest.raises(RuntimeError, match="shape"):
        CLIPEncoder().encode_text("cat")
