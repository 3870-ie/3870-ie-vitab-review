"""Frozen alternative vision encoders for the encoder comparison."""

from tempfile import NamedTemporaryFile
from pathlib import Path

import torch
from torch.nn import functional as F

from mmpfn.benchmarking.image_encoders import _as_pil


def extract_tokens(
    split, cache, model_path, vision, device, batch_size, patches_per_layer
):
    if vision == "clip":
        return split.get_embeddings(
            None,
            cache,
            batch_size,
            device,
            image_encoder="clip_vitl14",
            image_model_id=model_path,
            token_type="patch",
            clip_patch_layers=(6, 12, 18, 24),
        )
    width = 1024 if vision == "dinov2" else 768
    shape = (len(split.y), 1 + 4 * patches_per_layer, width)
    if cache.is_file():
        features = torch.load(cache, map_location="cpu", weights_only=True)
        if features.shape != shape:
            raise ValueError(f"Unexpected cache shape at {cache}")
        return features

    from transformers import AutoImageProcessor, AutoModel, CLIPModel

    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    if vision == "dinov2":
        processor.size = {"shortest_edge": 224}
        processor.crop_size = {"height": 224, "width": 224}
        model = AutoModel.from_pretrained(model_path, local_files_only=True)
    else:
        model = CLIPModel.from_pretrained(model_path, local_files_only=True)
    model.to(device).eval().requires_grad_(False)
    batches = []
    with torch.no_grad():
        for start in range(0, len(split.y), batch_size):
            paths = split.image_paths[start : start + batch_size]
            if any(len(row) != 1 for row in paths):
                raise ValueError(
                    "The encoder comparison requires exactly one image per row."
                )
            images = _as_pil([split._load_image(row[0], 518) for row in paths])
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(
                device
            )
            if vision == "dinov2":
                output = model(pixel_values=pixels, output_hidden_states=True)
                cls = output.last_hidden_state[:, :1]
                transform = model.layernorm
            else:
                output = model.vision_model(
                    pixel_values=pixels, output_hidden_states=True
                )
                cls = model.visual_projection(output.pooler_output).unsqueeze(1)

                def transform(value):
                    return model.visual_projection(
                        model.vision_model.post_layernorm(value)
                    )

            if len(output.hidden_states) != 25:
                raise ValueError("Expected a 24-block vision encoder.")
            layers = [cls]
            for layer in (6, 12, 18, 24):
                patches = output.hidden_states[layer][:, 1:]
                grid = patches.transpose(1, 2).reshape(len(paths), -1, 16, 16)
                layers.append(
                    transform(F.avg_pool2d(grid, 2).flatten(2).transpose(1, 2))
                )
            batches.append(torch.cat(layers, dim=1).cpu().float())
    features = torch.cat(batches)
    if features.shape != shape or not torch.isfinite(features).all():
        raise ValueError(f"Invalid features for {vision}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=cache.parent, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        torch.save(features, temporary)
        temporary.replace(cache)
    finally:
        temporary.unlink(missing_ok=True)
    return features
