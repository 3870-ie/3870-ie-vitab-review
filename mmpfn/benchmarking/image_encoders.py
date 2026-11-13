"""Frozen visual encoders shared by benchmark and encoder-comparison runs.

Original MMPFN paths cache one CLS token per selected image and optionally
project it to 768 dimensions. Patch Q-Former paths instead cache the ViT patch
sequence in FP16 and learn their own input/output projections. The confirmed
benchmark policy selects only the first image associated with each row.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from sklearn.decomposition import PCA


ImageEncoderName = Literal["dino_v2", "dino_v3", "clip_vitl14", "resnet50", "vit_b16"]
VisualTokenType = Literal["cls", "patch"]
ImageViewPolicy = Literal["first"]
ImagePathEntry = Path | Sequence[Path]
IMAGE_ENCODERS: tuple[ImageEncoderName, ...] = (
    "dino_v2",
    "dino_v3",
    "clip_vitl14",
    "resnet50",
    "vit_b16",
)

# The selected CLIP model, DINOv2-B and DINOv3-B all produce 768-dimensional
# vectors.  ResNet-50 produces 2048 dimensions and is reduced below.
DEFAULT_IMAGE_MODEL_IDS = {
    "dino_v3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "clip_vitl14": "openai/clip-vit-large-patch14",
}
MMPFN_IMAGE_DIM = 768


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location="cpu")


def _as_pil(images: Sequence[np.ndarray]) -> list[Image.Image]:
    converted: list[Image.Image] = []
    for image in images:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
        converted.append(Image.fromarray(array).convert("RGB"))
    return converted


def _extract_dino_v2(
    load_image: Callable[[Path, int], np.ndarray],
    paths: Sequence[Path],
    checkpoint: Path,
    batch_size: int,
    device: str,
    token_type: VisualTokenType,
) -> torch.Tensor:
    from mmpfn.models.dino_v2.models.vision_transformer import vit_base

    encoder = vit_base(
        patch_size=14,
        img_size=518,
        init_values=1.0,
        num_register_tokens=0,
        block_chunks=0,
    )
    encoder.load_state_dict(_torch_load(checkpoint))
    encoder = encoder.to(device).eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_images = np.stack([load_image(path, 336) for path in paths[start : start + batch_size]])
            batch = torch.from_numpy(np.moveaxis(batch_images, -1, 1)).to(device, non_blocking=True)
            encoded = encoder.forward_features(batch)
            features = encoded[
                "x_norm_clstoken" if token_type == "cls" else "x_norm_patchtokens"
            ]
            outputs.append(features.cpu())
    return torch.cat(outputs, dim=0)


def _extract_transformers(
    encoder_name: ImageEncoderName,
    load_image: Callable[[Path, int], np.ndarray],
    paths: Sequence[Path],
    model_id: str,
    batch_size: int,
    device: str,
    token_type: VisualTokenType,
    clip_layers: tuple[int, ...] | None = None,
    clip_patch_layers: tuple[int, ...] | None = None,
    clip_spatial_patch_layers: tuple[int, ...] | None = None,
) -> torch.Tensor:
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

    if encoder_name == "clip_vitl14":
        processor = CLIPProcessor.from_pretrained(model_id)
        encoder = CLIPModel.from_pretrained(model_id).to(device).eval()
    else:
        processor = AutoImageProcessor.from_pretrained(model_id)
        encoder = AutoModel.from_pretrained(model_id).to(device).eval()

    outputs = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            # The processor owns the model-specific resize, crop and
            # normalization.  Loading at 518 avoids an unnecessary loss of
            # detail before the processor applies its own transform.
            images = _as_pil([load_image(path, 518) for path in paths[start : start + batch_size]])
            pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(
                device, non_blocking=True
            )
            if encoder_name == "clip_vitl14":
                if clip_spatial_patch_layers is not None:
                    vision_output = encoder.vision_model(
                        pixel_values=pixel_values, output_hidden_states=True
                    )
                    hidden_states = vision_output.hidden_states
                    if hidden_states is None:
                        raise RuntimeError("CLIP did not return hidden states.")
                    n_blocks = len(hidden_states) - 1
                    if any(layer < 1 or layer > n_blocks for layer in clip_spatial_patch_layers):
                        raise ValueError(f"CLIP spatial patch layers exceed {n_blocks} blocks.")
                    cls = encoder.visual_projection(vision_output.pooler_output).unsqueeze(1)
                    patches = []
                    for layer in clip_spatial_patch_layers:
                        patch = hidden_states[layer][:, 1:, :]
                        side = int(patch.shape[1] ** 0.5)
                        if side != 16 or side * side != patch.shape[1]:
                            raise ValueError("Spatial CLIP expects a 16x16 patch grid.")
                        # 2x2 spatial average pooling keeps positions aligned.
                        patch = F.avg_pool2d(
                            patch.transpose(1, 2).reshape(patch.shape[0], patch.shape[2], 16, 16),
                            kernel_size=2, stride=2,
                        ).flatten(2).transpose(1, 2)
                        patches.append(encoder.visual_projection(encoder.vision_model.post_layernorm(patch)))
                    features = torch.cat((cls, *patches), dim=1)
                elif clip_patch_layers is not None:
                    vision_output = encoder.vision_model(
                        pixel_values=pixel_values, output_hidden_states=True
                    )
                    hidden_states = vision_output.hidden_states
                    if hidden_states is None:
                        raise RuntimeError("CLIP did not return hidden states.")
                    n_blocks = len(hidden_states) - 1
                    if any(layer < 1 or layer > n_blocks for layer in clip_patch_layers):
                        raise ValueError(
                            f"CLIP patch layers must be in [1, {n_blocks}], "
                            f"got {clip_patch_layers}."
                        )
                    post_ln = encoder.vision_model.post_layernorm
                    projection = encoder.visual_projection
                    cls = projection(vision_output.pooler_output).unsqueeze(1)
                    patches = [
                        projection(post_ln(hidden_states[layer][:, 1:, :]))
                        for layer in clip_patch_layers
                    ]
                    features = torch.cat((cls, *patches), dim=1)
                elif clip_layers is not None:
                    vision_output = encoder.vision_model(
                        pixel_values=pixel_values, output_hidden_states=True
                    )
                    hidden_states = vision_output.hidden_states
                    if hidden_states is None:
                        raise RuntimeError("CLIP did not return hidden states.")
                    n_blocks = len(hidden_states) - 1
                    if any(layer < 1 or layer > n_blocks for layer in clip_layers):
                        raise ValueError(
                            f"CLIP layers must be in [1, {n_blocks}], got {clip_layers}."
                        )
                    # hidden_states[0] is the input embedding; state[k] is the
                    # output of block k. The final state follows the same
                    # post-LN + projection path as the historical CLS cache.
                    features = torch.stack(
                        [
                            encoder.visual_projection(
                                encoder.vision_model.post_layernorm(
                                    hidden_states[layer][:, 0, :]
                                )
                            )
                            for layer in clip_layers
                        ],
                        dim=1,
                    )
                elif token_type == "cls":
                    # Transformers has changed ``CLIPModel.get_image_features``
                    # across releases: some versions return the projected tensor,
                    # while newer variants can return ``BaseModelOutputWithPooling``.
                    # Run the two documented CLIP submodules explicitly so this
                    # adapter always emits the projected CLIP image embedding
                    # (768-d for ViT-L/14), rather than depending on that API
                    # return-type detail.
                    vision_output = encoder.vision_model(pixel_values=pixel_values)
                    pooled = getattr(vision_output, "pooler_output", None)
                    if pooled is None:
                        pooled = vision_output.last_hidden_state[:, 0, :]
                    features = encoder.visual_projection(pooled)
                else:
                    # Keep the spatial ViT tokens in the vision-transformer
                    # width.  The Q-Former owns the learned input projection.
                    features = encoder.vision_model(
                        pixel_values=pixel_values
                    ).last_hidden_state[:, 1:, :]
            else:
                model_output = encoder(pixel_values=pixel_values)
                if token_type == "cls":
                    features = model_output.last_hidden_state[:, 0, :]
                else:
                    register_tokens = int(
                        getattr(getattr(encoder, "config", None), "num_register_tokens", 0)
                    )
                    features = model_output.last_hidden_state[:, 1 + register_tokens :, :]
            outputs.append(features.cpu().half() if clip_patch_layers is not None or clip_spatial_patch_layers is not None else features.cpu())
    return torch.cat(outputs, dim=0)


def _extract_torchvision(
    encoder_name: ImageEncoderName,
    load_image: Callable[[Path, int], np.ndarray],
    paths: Sequence[Path],
    batch_size: int,
    device: str,
    token_type: VisualTokenType,
) -> torch.Tensor:
    from torchvision.models import (
        ResNet50_Weights,
        ViT_B_16_Weights,
        resnet50,
        vit_b_16,
    )

    if encoder_name == "resnet50":
        if token_type == "patch":
            raise ValueError(
                "Patch Q-Former intentionally has no ResNet adapter; choose a ViT encoder."
            )
        weights = ResNet50_Weights.IMAGENET1K_V2
        encoder = resnet50(weights=weights)
        encoder.fc = torch.nn.Identity()
    else:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        encoder = vit_b_16(weights=weights)
        encoder.heads = torch.nn.Identity()
    transform = weights.transforms()
    encoder = encoder.to(device).eval()

    outputs = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            images = np.stack([load_image(path, 518) for path in paths[start : start + batch_size]])
            batch = torch.from_numpy(np.moveaxis(images, -1, 1))
            # torchvision's official weight transform applies resize/crop and
            # ImageNet normalization to the whole tensor batch.
            transformed = transform(batch).to(device, non_blocking=True)
            if token_type == "cls":
                features = encoder(transformed)
            else:
                # Mirrors torchvision VisionTransformer.forward but retains
                # its spatial sequence instead of applying the classifier.
                patches = encoder._process_input(transformed)
                class_token = encoder.class_token.expand(patches.shape[0], -1, -1)
                features = encoder.encoder(torch.cat((class_token, patches), dim=1))[:, 1:, :]
            outputs.append(features.cpu())
    return torch.cat(outputs, dim=0)


def extract_image_embeddings(
    *,
    encoder_name: ImageEncoderName,
    image_paths: Sequence[ImagePathEntry],
    load_image: Callable[[Path, int], np.ndarray],
    cache_path: str | Path,
    dino_checkpoint: str | Path | None,
    image_model_id: str | None,
    batch_size: int = 16,
    device: str = "cuda",
    token_type: VisualTokenType = "cls",
    image_view_policy: ImageViewPolicy = "first",
    clip_layers: Sequence[int] | None = None,
    clip_patch_layers: Sequence[int] | None = None,
    clip_spatial_patch_layers: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return frozen CLS/view tokens or patch tokens for every example.

    ``token_type='cls'`` returns ``[N, 1, D]`` and preserves the original
    MMPFN input convention. ``token_type='patch'`` returns ``[N, P, D]`` for
    Patch Q-Former. When ``clip_layers`` is set, the token axis holds one
    projected CLS vector per requested CLIP block. The confirmed VT-Bench
    policy selects the first image before extraction.
    """
    if token_type not in {"cls", "patch"}:
        raise ValueError(f"Unknown visual token type: {token_type}")
    clip_layers = tuple(clip_layers) if clip_layers is not None else None
    clip_patch_layers = (
        tuple(clip_patch_layers) if clip_patch_layers is not None else None
    )
    clip_spatial_patch_layers = (
        tuple(clip_spatial_patch_layers) if clip_spatial_patch_layers is not None else None
    )
    if clip_spatial_patch_layers is not None:
        if (clip_layers is not None or clip_patch_layers is not None or
                encoder_name != "clip_vitl14" or token_type != "patch"):
            raise ValueError("Spatial CLIP requires CLIP patch extraction without legacy layer options.")
        if clip_spatial_patch_layers not in {(24,), (6, 12, 18, 24)}:
            raise ValueError("Spatial CLIP supports layer 24 or layers 6/12/18/24.")
    if clip_patch_layers is not None:
        if clip_layers is not None or encoder_name != "clip_vitl14" or token_type != "patch":
            raise ValueError("clip_patch_layers requires CLIP patch extraction without clip_layers.")
        if clip_patch_layers not in {(24,), (6, 12, 18, 24)}:
            raise ValueError("Supported CLIP patch layers are (24,) or (6, 12, 18, 24).")
    if clip_layers is not None:
        if encoder_name != "clip_vitl14" or token_type != "cls":
            raise ValueError("clip_layers requires clip_vitl14 CLS extraction.")
        if not clip_layers or len(set(clip_layers)) != len(clip_layers):
            raise ValueError("clip_layers must contain distinct block numbers.")
    if image_view_policy != "first":
        raise ValueError("Only the confirmed image_view_policy='first' is implemented.")
    cache_path = Path(cache_path)
    path_sets = [
        (entry,) if isinstance(entry, Path) else tuple(entry)
        for entry in image_paths
    ]
    if not path_sets:
        raise ValueError("Cannot extract image embeddings from an empty split.")
    if any(len(paths) == 0 for paths in path_sets):
        raise ValueError("Every example must provide at least one image path.")
    # MMPFN/VT-Bench evaluates only the first image of a multi-image example.
    flat_paths = [paths[0] for paths in path_sets]
    if cache_path.is_file():
        embeddings = _torch_load(cache_path)
        if len(embeddings) != len(path_sets) or embeddings.ndim != 3:
            raise ValueError(
                f"Embedding cache does not match {cache_path}: got {tuple(embeddings.shape)}, "
                f"expected ({len(path_sets)}, tokens, dim)."
            )
        if clip_patch_layers is not None and embeddings.shape[1:] != (
            1 + 256 * len(clip_patch_layers), 768
        ):
            raise ValueError(f"CLIP CLS+Patch cache has unexpected shape: {cache_path}.")
        if clip_spatial_patch_layers is not None and embeddings.shape[1:] != (
            1 + 64 * len(clip_spatial_patch_layers), 768
        ):
            raise ValueError(f"Spatial CLIP cache has unexpected shape: {cache_path}.")
        if clip_layers is not None and embeddings.shape[1] != len(clip_layers):
            raise ValueError(
                f"Embedding cache {cache_path} has {embeddings.shape[1]} layers; "
                f"expected {len(clip_layers)}."
            )
        return embeddings if clip_patch_layers is not None or clip_spatial_patch_layers is not None else embeddings.float()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Visual embedding extraction requires CUDA, but CUDA is unavailable.")
    if encoder_name == "dino_v2":
        if dino_checkpoint is None or not Path(dino_checkpoint).is_file():
            raise FileNotFoundError("dino_v2 requires --dino-checkpoint pointing to dinov2_vitb14_pretrain.pth.")
        raw = _extract_dino_v2(
            load_image, flat_paths, Path(dino_checkpoint), batch_size, device, token_type
        )
    elif encoder_name in {"dino_v3", "clip_vitl14"}:
        model_id = image_model_id or DEFAULT_IMAGE_MODEL_IDS[encoder_name]
        raw = _extract_transformers(
            encoder_name, load_image, flat_paths, model_id, batch_size, device,
            token_type, clip_layers, clip_patch_layers, clip_spatial_patch_layers,
        )
    elif encoder_name in {"resnet50", "vit_b16"}:
        raw = _extract_torchvision(
            encoder_name, load_image, flat_paths, batch_size, device, token_type
        )
    else:  # defensive guard for CLI values passed programmatically
        raise ValueError(f"Unsupported image encoder: {encoder_name}")

    embeddings = raw if clip_patch_layers is not None or clip_spatial_patch_layers is not None else raw.float()
    if token_type == "cls" and clip_layers is None and clip_spatial_patch_layers is None:
        embeddings = embeddings.unsqueeze(1)
    if embeddings.ndim != 3 or len(embeddings) != len(path_sets):
        raise RuntimeError(
            f"Encoder returned invalid {token_type} features: {tuple(embeddings.shape)}"
        )
    if clip_layers is not None and embeddings.shape[1] != len(clip_layers):
        raise RuntimeError(
            f"CLIP returned {embeddings.shape[1]} layers, expected {len(clip_layers)}."
        )
    if clip_patch_layers is not None and embeddings.shape[1:] != (
        1 + 256 * len(clip_patch_layers), 768
    ):
        raise RuntimeError("CLIP CLS+Patch extraction returned an unexpected shape.")
    if clip_spatial_patch_layers is not None and embeddings.shape[1:] != (
        1 + 64 * len(clip_spatial_patch_layers), 768
    ):
        raise RuntimeError("Spatial CLIP extraction returned an unexpected shape.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Patch caches dominate disk usage; FP16 is sufficient for a frozen
    # encoder cache and is converted to the runner's compute dtype on load.
    cached = embeddings.half() if token_type == "patch" else embeddings
    with NamedTemporaryFile(dir=cache_path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(cached, temporary_path)
        temporary_path.replace(cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return cached if clip_patch_layers is not None or clip_spatial_patch_layers is not None else embeddings


def project_for_mmpfn(
    embeddings: dict[str, torch.Tensor],
    projection_path: str | Path,
    *,
    output_dim: int = MMPFN_IMAGE_DIM,
    max_fit_rows: int = 10_000,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Project raw frozen features to MMPFN's 768-dimensional token space.

    The projector is fitted from ``embeddings['train']`` only.  ``identity``
    preserves the historical DINOv2 path exactly; ``pca`` is used for wider
    encoders such as ResNet-50; and ``pad`` is a deterministic fallback for a
    narrower model.
    """
    projection_path = Path(projection_path)
    train = embeddings["train"].detach().cpu().float()
    if train.ndim != 3:
        raise ValueError(f"Expected [N, n_views, D] embeddings, got {tuple(train.shape)}")
    input_dim = int(train.shape[-1])

    if projection_path.is_file():
        state = _torch_load(projection_path)
        if int(state["input_dim"]) != input_dim or int(state["output_dim"]) != output_dim:
            raise ValueError(f"Projection cache is incompatible with current embeddings: {projection_path}")
    elif input_dim == output_dim:
        state = {"method": "identity", "input_dim": input_dim, "output_dim": output_dim}
        projection_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, projection_path)
    elif input_dim < output_dim:
        state = {"method": "pad", "input_dim": input_dim, "output_dim": output_dim}
        projection_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, projection_path)
    else:
        values = train.reshape(-1, input_dim).numpy()
        if len(values) > max_fit_rows:
            rng = np.random.default_rng(seed)
            values = values[rng.choice(len(values), size=max_fit_rows, replace=False)]
        # A tiny smoke-test split can contain fewer than 768 rows.  PCA then
        # has fewer usable axes; retain all it has and zero-pad afterwards.
        pca_dim = min(output_dim, input_dim, len(values))
        pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=seed)
        pca.fit(values)
        state = {
            "method": "pca",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "mean": torch.from_numpy(pca.mean_.astype(np.float32)),
            "components": torch.from_numpy(pca.components_.astype(np.float32)),
            "fit_rows": int(len(values)),
            "pca_dim": int(pca_dim),
        }
        projection_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, projection_path)

    method = state["method"]
    projected: dict[str, torch.Tensor] = {}
    for split, value in embeddings.items():
        tokens = value.detach().cpu().float()
        if tokens.ndim != 3 or tokens.shape[-1] != input_dim:
            raise ValueError(f"Expected [N, n_views, {input_dim}] embeddings for {split}, got {tuple(tokens.shape)}")
        matrix = tokens.reshape(-1, input_dim)
        if method == "identity":
            result = matrix
        elif method == "pad":
            result = torch.nn.functional.pad(matrix, (0, output_dim - input_dim))
        elif method == "pca":
            result = (matrix - state["mean"]) @ state["components"].T
            if result.shape[1] < output_dim:
                result = torch.nn.functional.pad(result, (0, output_dim - result.shape[1]))
        else:
            raise ValueError(f"Unknown projection method in {projection_path}: {method}")
        projected[split] = result.reshape(tokens.shape[0], tokens.shape[1], output_dim).contiguous()
    return projected
