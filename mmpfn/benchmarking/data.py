"""Unified prepared-split loader used by the benchmark runner."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from mmpfn.benchmarking.image_encoders import (
    ImageEncoderName,
    ImageViewPolicy,
    VisualTokenType,
    extract_image_embeddings,
)
from mmpfn.benchmarking.registry import DatasetSpec
from mmpfn.datasets.vtbench import VTBenchSplitDataset


class PreparedBenchmarkSplit:
    """A portable ``train/val/test.npz`` image-tabular split."""

    def __init__(self, root: str | Path, spec: DatasetSpec, split: str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.spec = spec
        self.dataset = spec.key
        self.split = split
        metadata_path = self.root / "metadata.json"
        split_path = self.root / f"{split}.npz"
        if not metadata_path.is_file() or not split_path.is_file():
            raise FileNotFoundError(
                f"Prepared dataset is incomplete at {self.root}; expected metadata.json and {split}.npz."
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(split_path, allow_pickle=False) as payload:
            self.x = np.asarray(payload["x"], dtype=np.float32)
            label_dtype = np.int64 if spec.task == "classification" else np.float32
            self.y = np.asarray(payload["y"], dtype=label_dtype).reshape(-1)
            if spec.secondary_modality == "image":
                raw_paths = np.asarray(payload["image_paths"]).astype(str)
                raw_texts: list[str] = []
            else:
                raw_paths = []
                raw_texts = np.asarray(payload["texts"]).astype(str).tolist()

        image_base = Path(self.metadata.get("image_base_dir", self.root)).expanduser()
        if not image_base.is_absolute():
            image_base = (self.root / image_base).resolve()
        if spec.secondary_modality == "image":
            if raw_paths.ndim == 1:
                raw_path_sets = [[path] for path in raw_paths.tolist()]
            elif raw_paths.ndim == 2:
                raw_path_sets = raw_paths.tolist()
            else:
                raise ValueError(
                    f"{spec.key}/{split} image_paths must be one or two dimensional, got {raw_paths.shape}."
                )
            self.image_paths = [
                [Path(path) if Path(path).is_absolute() else (image_base / path).resolve() for path in path_set]
                for path_set in raw_path_sets
            ]
        else:
            self.image_paths = []
        self.texts = raw_texts
        self.image_encoding = self.metadata.get("image_encoding", spec.image_encoding)
        self.categorical_features = [int(index) for index in self.metadata.get("categorical_indices", [])]
        self.field_lengths = self.metadata.get("field_lengths")
        self.embeddings: torch.Tensor | None = None

        secondary_count = len(self.image_paths) if spec.secondary_modality == "image" else len(self.texts)
        if len(self.x) != len(self.y) or len(self.x) != secondary_count:
            raise ValueError(
                f"Misaligned {spec.key}/{split}: x={len(self.x)}, y={len(self.y)}, secondary={secondary_count}"
            )
        if spec.secondary_modality == "image":
            missing = [path for path_set in self.image_paths for path in path_set if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{spec.key}/{split} references {len(missing)} missing images; first: {missing[0]}"
                )

    def _load_image(self, path: Path, image_size: int) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            image = np.asarray(np.load(path), dtype=np.float32)
            if image.ndim == 2:
                image = image[:, :, None]
            if image.ndim != 3:
                raise ValueError(f"Expected 2D/3D image array, got {image.shape}: {path}")
            if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
                image = np.moveaxis(image, 0, -1)
            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            if image.shape[-1] != 3:
                raise ValueError(f"Expected one or three channels, got {image.shape}: {path}")
            if self.image_encoding == "imagenet_normalized":
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                image = image * std + mean
            elif self.image_encoding == "uint8":
                image = image / 255.0
            image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
            pil_image = Image.fromarray(image)
        else:
            with Image.open(path) as opened:
                pil_image = ImageOps.exif_transpose(opened).convert("RGB")
        resized = pil_image.resize(
            (image_size, image_size),
            getattr(Image, "Resampling", Image).BILINEAR,
        )
        return np.asarray(resized, dtype=np.float32) / 255.0

    def get_embeddings(
        self,
        dino_checkpoint: str | Path | None,
        cache_path: str | Path,
        batch_size: int = 16,
        device: str = "cuda",
        image_encoder: ImageEncoderName = "dino_v2",
        image_model_id: str | None = None,
        token_type: VisualTokenType = "cls",
        image_view_policy: ImageViewPolicy = "first",
        clip_layers: Sequence[int] | None = None,
        clip_patch_layers: Sequence[int] | None = None,
        clip_spatial_patch_layers: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if self.spec.secondary_modality == "text":
            self.embeddings = self._get_text_embeddings(Path(cache_path), batch_size, device)
            return self.embeddings
        self.embeddings = extract_image_embeddings(
            encoder_name=image_encoder,
            image_paths=self.image_paths,
            load_image=self._load_image,
            cache_path=cache_path,
            dino_checkpoint=dino_checkpoint,
            image_model_id=image_model_id,
            batch_size=batch_size,
            device=device,
            token_type=token_type,
            image_view_policy=image_view_policy,
            clip_layers=clip_layers,
            clip_patch_layers=clip_patch_layers,
            clip_spatial_patch_layers=clip_spatial_patch_layers,
        )
        return self.embeddings

    def _get_text_embeddings(self, cache_path: Path, batch_size: int, device: str) -> torch.Tensor:
        """Embed the original MMPFN paper text input with its Electra encoder."""
        model_id = self.metadata.get("text_model_id") or self.spec.text_model_id
        if not model_id:
            raise ValueError(f"No text encoder configured for {self.dataset}.")
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        encoder = AutoModel.from_pretrained(model_id).to(device).eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(self.texts), batch_size):
                batch_text = self.texts[start : start + batch_size]
                tokens = tokenizer(
                    batch_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                tokens = {name: value.to(device, non_blocking=True) for name, value in tokens.items()}
                features = encoder(**tokens).last_hidden_state[:, 0, :]
                outputs.append(features.unsqueeze(1).cpu())
        self.embeddings = torch.cat(outputs, dim=0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.embeddings, cache_path)
        return self.embeddings


def load_benchmark_splits(spec: DatasetSpec, root: str | Path) -> dict[str, Any]:
    """Load portable prepared files, with a compatibility path for current VT exports."""
    root = Path(root).expanduser().resolve()
    if (root / "metadata.json").is_file() and (root / "train.npz").is_file():
        return {split: PreparedBenchmarkSplit(root, spec, split) for split in ("train", "val", "test")}
    if spec.benchmark == "vtbench" and spec.legacy_vtbench_name:
        return {
            split: VTBenchSplitDataset(
                root,
                spec.legacy_vtbench_name,
                split,
                spec.image_encoding,
            )
            for split in ("train", "val", "test")
        }
    raise FileNotFoundError(
        f"No prepared data found for {spec.key} at {root}. "
        "Expected metadata.json plus train.npz/val.npz/test.npz."
    )
