"""Visual Pseudo-Column Fusion on official TabICLv2.

The paper workflow is exposed through `python -m vitab_adapter run`.
This module provides the per-candidate training and evaluation routines.

Table features use TabICLv2's official TransformToNumerical + EnsembleGenerator
preprocessing. Visual pseudo-columns remain differentiable, use context-only
z-score normalization, and are concatenated before a joint feature permutation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch import nn
from tabicl._finetune.regressor import _pinball_loss as tabicl_pinball_loss
from tabicl.train._optim import get_scheduler as tabicl_get_scheduler

from mmpfn.benchmarking.data import load_benchmark_splits
from mmpfn.benchmarking.registry import get_dataset_spec


PRIMARY_PROTOCOL_VERSION = "accuracy_rmse_v1"


def get_primary_protocol(task: str) -> tuple[str, bool]:
    if task == "classification":
        return "accuracy", True
    if task == "regression":
        return "rmse", False
    raise ValueError(f"Unsupported task for primary metric protocol: {task!r}")


# -----------------------------------------------------------------------------
# Projectors
# -----------------------------------------------------------------------------


class VisualPseudoColumnMLP(nn.Module):
    def __init__(self, visual_dim: int, pseudo_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pseudo_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiLayerCLIPFusion(nn.Module):
    """Task-level layer/branch weights plus sample-dependent spatial attention pooling."""

    patch_layers = (6, 12, 18, 24)
    patches_per_layer = 64

    def __init__(self, visual_dim: int, attention_hidden_dim: int = 128):
        super().__init__()
        self.variant = "E3"
        self.layer_logits = nn.Parameter(torch.zeros(4))
        self.branch_logits = nn.Parameter(torch.zeros(2))  # CLS, Patch
        self.spatial_score = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # [rows, views, CLS+4*64, D] -> [rows, D]. Match the original
        # pseudo_mlp's mean-over-views behavior before visual fusion.
        if image.ndim != 4 or image.shape[2] != 1 + 4 * self.patches_per_layer:
            raise ValueError(f"Unexpected multi-layer token shape: {tuple(image.shape)}")
        tokens = image.mean(dim=1)
        cls = tokens[:, 0, :]
        patches = tokens[:, 1:, :].reshape(
            tokens.shape[0], 4, self.patches_per_layer, tokens.shape[-1]
        )
        if self.variant == "E1":
            patches = patches.reshape(len(tokens), 4, 8, 8, tokens.shape[-1]).mean(dim=(2, 3)).unsqueeze(2)
        elif self.variant == "E2":
            patches = patches.reshape(len(tokens), 4, 4, 2, 4, 2, tokens.shape[-1]).mean(dim=(3, 5)).flatten(2, 3)
        layer_weights = self.layer_logits.softmax(dim=0)
        fused_patches = (patches * layer_weights[None, :, None, None]).sum(dim=1)
        spatial_weights = self.spatial_score(fused_patches).squeeze(-1).softmax(dim=-1)
        patch_vector = (fused_patches * spatial_weights.unsqueeze(-1)).sum(dim=1)
        branch_weights = self.branch_logits.softmax(dim=0)
        return branch_weights[0] * cls + branch_weights[1] * patch_vector

    def weight_snapshot(self) -> dict[str, float]:
        layer = self.layer_logits.detach().float().softmax(dim=0).cpu().tolist()
        branch = self.branch_logits.detach().float().softmax(dim=0).cpu().tolist()
        return {
            **{f"patch_layer_{l}": float(w) for l, w in zip(self.patch_layers, layer)},
            "cls_weight": float(branch[0]),
            "patch_weight": float(branch[1]),
        }


class CrossAttentionPooler(nn.Module):
    """MMPFN-style CAP with K output queries and a fixed attention-head count."""

    def __init__(self, dim: int, num_output_tokens: int, attention_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if dim % attention_heads != 0:
            raise ValueError(f"attention_heads={attention_heads} must divide dim={dim}")
        self.queries = nn.Parameter(torch.randn(num_output_tokens, dim) * 1e-2)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.attn = nn.MultiheadAttention(dim, attention_heads, dropout=dropout, batch_first=True)
        self.k_norm = nn.LayerNorm(dim)
        self.q_norm = nn.LayerNorm(dim)
        self.out_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: [B, rows, visual_tokens, dim]
        if src.ndim != 4:
            raise ValueError(f"CAP expects [B, rows, tokens, dim], got {tuple(src.shape)}")
        B, T, _, D = src.shape
        kv = self.k_norm(src).reshape(B * T, -1, D)
        q = self.q_proj(self.q_norm(self.queries)).unsqueeze(0).expand(B * T, -1, -1)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        out = self.out_norm(out) + self.ffn(out)
        return out.reshape(B, T, -1, D)


class TabICLPseudoColumnModel(nn.Module):
    def __init__(
        self,
        tabicl: nn.Module,
        visual_dim: int,
        fusion_mode: str,
        pseudo_dim: int,
        mgm_heads: int = 8,
        cap_attention_heads: int = 8,
        dropout: float = 0.1,
        multilayer_clip: bool = False,
        tabular_shortcut: bool = False,
    ):
        super().__init__()
        if fusion_mode not in {"pseudo_mlp", "pseudo_mmpfn"}:
            raise ValueError("fusion_mode must be pseudo_mlp or pseudo_mmpfn")
        if multilayer_clip and fusion_mode != "pseudo_mlp":
            raise ValueError("Multi-layer CLIP is only supported by the pseudo_mlp main path")
        self.tabicl = tabicl
        self.tabular_shortcut = tabular_shortcut
        self.tabicl_t = copy.deepcopy(tabicl) if tabular_shortcut else None
        self.table_weight = nn.Parameter(torch.zeros(())) if tabular_shortcut else None
        self.fusion_mode = fusion_mode
        self.pseudo_dim = pseudo_dim
        self.multilayer_clip = multilayer_clip
        self._freeze_tabicl = True
        self._freeze_pseudo = False
        self.phase2_mode: str | None = None
        self._debug_printed = False

        self.visual_pseudo_mlp = None
        self.visual_multilayer_fusion = None
        self.visual_mgm = None
        self.visual_cap = None
        self.visual_scalar_head = None
        if fusion_mode == "pseudo_mlp":
            self.visual_pseudo_mlp = VisualPseudoColumnMLP(visual_dim, pseudo_dim, dropout=dropout)
            if multilayer_clip:
                self.visual_multilayer_fusion = MultiLayerCLIPFusion(visual_dim)
        else:
            raise ValueError('This release supports the pseudo_mlp adapter.')

    def pseudo_parameters(self) -> Iterator[nn.Parameter]:
        for module in (
            self.visual_pseudo_mlp, self.visual_multilayer_fusion,
            self.visual_mgm, self.visual_cap, self.visual_scalar_head,
        ):
            if module is not None:
                yield from module.parameters()

    def configure_phase(self, phase: int, phase2_mode: str | None = None) -> None:
        if self.tabular_shortcut and phase != 1:
            raise ValueError("Tabular shortcut is only supported in frozen-TabICL Phase 1")
        self.requires_grad_(False)
        self.phase2_mode = phase2_mode if phase == 2 else None
        self._freeze_tabicl = phase == 1
        self._freeze_pseudo = phase == 2 and phase2_mode == "backbone_only"
        if phase == 1 or phase2_mode == "joint":
            for p in self.pseudo_parameters():
                p.requires_grad_(True)
        if self.tabular_shortcut:
            self.table_weight.requires_grad_(True)
        if phase == 2:
            for p in self.tabicl.parameters():
                p.requires_grad_(True)
        self.train(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_tabicl:
            self.tabicl.eval()
        if self.tabicl_t is not None:
            self.tabicl_t.eval()
        if self._freeze_pseudo:
            for module in (
                self.visual_pseudo_mlp, self.visual_multilayer_fusion,
                self.visual_mgm, self.visual_cap, self.visual_scalar_head,
            ):
                if module is not None:
                    module.eval()
        return self

    def _pseudo_columns(self, image: torch.Tensor, device: torch.device) -> torch.Tensor:
        # Single-vector image: [rows, views, D]. Multi-layer CLIP: [rows, views, 257, D].
        x = image.unsqueeze(0)
        if self.fusion_mode == "pseudo_mlp":
            assert self.visual_pseudo_mlp is not None
            if self.multilayer_clip:
                assert self.visual_multilayer_fusion is not None
                if not self.training and (image.shape[0] > 256 or image.device != device):
                    # Evaluation retains the complete context. Projecting its
                    # CLIP tokens in chunks gives the same row-wise values.
                    # Keep the large raw token batch on CPU and transfer only
                    # one chunk at a time; otherwise validation can allocate
                    # more than a 32-GiB GPU before the projector runs.
                    out = torch.cat(
                        [self.visual_pseudo_mlp(self.visual_multilayer_fusion(part.to(device, non_blocking=True)))
                         for part in image.split(256, dim=0)], dim=0
                    ).unsqueeze(0)
                else:
                    out = self.visual_pseudo_mlp(self.visual_multilayer_fusion(image)).unsqueeze(0)
            else:
                out = self.visual_pseudo_mlp(x.mean(dim=2))
        else:
            assert self.visual_mgm is not None and self.visual_cap is not None
            assert self.visual_scalar_head is not None
            out = self.visual_scalar_head(self.visual_cap(self.visual_mgm(x))).squeeze(-1)
        return out[0]

    @staticmethod
    def _zscore_from_context(pseudo: torch.Tensor, context_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = pseudo[:context_size]
        mean = context.mean(0, keepdim=True)
        std = context.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
        return (pseudo - mean) / std, mean, std

    def _combine(
        self,
        table: torch.Tensor,
        image: torch.Tensor,
        context_size: int,
        joint_shuffles: torch.Tensor,
        raw_table_shape: tuple[int, int | None] | None = None,
        precomputed_pseudo: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # table: [E, rows, H], image: [rows, views, D]
        raw_pseudo = precomputed_pseudo if precomputed_pseudo is not None else self._pseudo_columns(image, table.device)
        if raw_pseudo.shape[0] != table.shape[1]:
            raise ValueError(f"Pseudo-column rows {raw_pseudo.shape[0]} != table rows {table.shape[1]}")
        pseudo_norm, pseudo_mu, pseudo_std = self._zscore_from_context(raw_pseudo, context_size)
        pseudo = pseudo_norm.unsqueeze(0).expand(table.shape[0], -1, -1).to(table.dtype)
        X = torch.cat([table, pseudo], dim=-1)
        gather = joint_shuffles[:, None, :].expand(-1, X.shape[1], -1)
        X = torch.gather(X, -1, gather)
        mode = getattr(self, "modality", "full")
        if mode != "full":
            visual = joint_shuffles >= table.shape[-1]
            mask = visual if mode == "without-image" else ~visual
            X = torch.where(mask[:, None, :], X[:, :context_size].mean(dim=1, keepdim=True), X)
        if not self._debug_printed:
            print(
                "Pseudo-column debug: "
                f"raw_table={raw_table_shape} "
                f"processed_table={tuple(table.shape)} "
                f"raw_visual={tuple(image.shape)} "
                f"pseudo={tuple(raw_pseudo.shape)} "
                f"normalized_pseudo={tuple(pseudo.shape)} "
                f"final={tuple(X.shape)} "
                f"pseudo_mu_minmax=({float(pseudo_mu.min().detach().cpu()):.6g},"
                f"{float(pseudo_mu.max().detach().cpu()):.6g}) "
                f"pseudo_std_minmax=({float(pseudo_std.min().detach().cpu()):.6g},"
                f"{float(pseudo_std.max().detach().cpu()):.6g})",
                flush=True,
            )
            self._debug_printed = True
        return X

    def training_outputs(self, batch: "MetaBatch", *, classification: bool = True) -> torch.Tensor:
        offload = getattr(
            self.visual_multilayer_fusion, "variant", None
        ) == "E4"
        if offload and batch.table.is_cuda:
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                return self._training_outputs(batch, classification=classification)
        return self._training_outputs(batch, classification=classification)

    def _training_outputs(self, batch: "MetaBatch", *, classification: bool) -> torch.Tensor:
        X = self._combine(batch.table, batch.image, batch.context_size, batch.joint_shuffles, batch.raw_table_shape)
        y_train = batch.y_context.float()
        train_size = y_train.shape[1]
        # Keep the frozen TabICLv2 backbone in eval mode during Phase 1, but
        # explicitly call its differentiable training-path building blocks.
        embeddings = self.tabicl.col_embedder._train_forward(X, y_train=y_train, d=None, embed_with_test=False)
        representations = self.tabicl.row_interactor._train_forward(embeddings, d=None)
        out = self.tabicl.icl_predictor._icl_predictions(representations, y_train)
        vt = out[:, train_size:]
        if self.tabicl_t is None:
            return vt
        with torch.no_grad():
            table_embeddings = self.tabicl_t.col_embedder._train_forward(
                batch.table, y_train=y_train, d=None, embed_with_test=False
            )
            table_representations = self.tabicl_t.row_interactor._train_forward(table_embeddings, d=None)
            table = self.tabicl_t.icl_predictor._icl_predictions(table_representations, y_train)[:, train_size:]
        if not classification:
            vt = self.tabicl.quantile_dist(vt).quantiles.mean(dim=-1)
            table = self.tabicl_t.quantile_dist(table).quantiles.mean(dim=-1)
        final = (1 - self.table_weight) * vt + self.table_weight * table
        return final, vt, table

    def inference_outputs(
        self, batch: "MetaBatch", classification: bool,
        precomputed_pseudo: torch.Tensor | None = None,
    ) -> torch.Tensor:
        X = self._combine(
            batch.table, batch.image, batch.context_size, batch.joint_shuffles,
            batch.raw_table_shape, precomputed_pseudo=precomputed_pseudo,
        )
        if classification:
            vt = self.tabicl(X=X, y_train=batch.y_context.float(), return_logits=True)
            if self.tabicl_t is None:
                return vt
            table = self.tabicl_t(X=batch.table, y_train=batch.y_context.float(), return_logits=True)
            return (1 - self.table_weight) * vt + self.table_weight * table, vt, table
        raw = self.tabicl(X=X, y_train=batch.y_context.float())
        vt = self.tabicl.quantile_dist(raw).quantiles.mean(dim=-1)
        if self.tabicl_t is None:
            return vt
        table_raw = self.tabicl_t(X=batch.table, y_train=batch.y_context.float())
        table = self.tabicl_t.quantile_dist(table_raw).quantiles.mean(dim=-1)
        return (1 - self.table_weight) * vt + self.table_weight * table, vt, table


# -----------------------------------------------------------------------------
# Official TabICLv2 preprocessing + aligned image batches
# -----------------------------------------------------------------------------


@dataclass
class MetaBatch:
    table: torch.Tensor
    image: torch.Tensor
    y_context: torch.Tensor
    y_query: torch.Tensor
    joint_shuffles: torch.Tensor
    class_shuffles: list[np.ndarray] | None
    context_size: int
    raw_table_shape: tuple[int, int | None] | None = None
    y_mean: float | None = None
    y_std: float | None = None


def _pp_classes():
    from tabicl._sklearn.preprocessing import EnsembleGenerator, Shuffler, TransformToNumerical

    return EnsembleGenerator, Shuffler, TransformToNumerical


def _patterns(shuffler, n: int) -> list[np.ndarray]:
    raw = shuffler.shuffle(n)
    if not raw:
        raise RuntimeError("TabICL Shuffler returned no patterns")
    return [np.asarray(raw[i % len(raw)], dtype=np.int64) for i in range(n)]


def _split(y: np.ndarray, query_size: int, seed: int, classification: bool):
    query_size = min(max(1, query_size), len(y) - 1)
    if classification:
        query_size = min(max(query_size, len(np.unique(y))), len(y) - 1)
        try:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=query_size, random_state=seed)
            return next(splitter.split(np.zeros((len(y), 1)), y))
        except ValueError:
            rng = np.random.default_rng(seed)
            query_parts = []
            for label in np.unique(y):
                label_idx = np.flatnonzero(y == label)
                if len(label_idx) > 1:
                    query_parts.append(rng.choice(label_idx, size=1, replace=False))
            if query_parts:
                query_idx = np.concatenate(query_parts)
            else:
                query_idx = rng.choice(len(y), size=1, replace=False)
            remaining = np.setdiff1d(np.arange(len(y)), query_idx, assume_unique=False)
            extra = min(max(0, query_size - len(query_idx)), len(remaining))
            if extra:
                query_idx = np.concatenate([query_idx, rng.choice(remaining, size=extra, replace=False)])
            train_idx = np.setdiff1d(np.arange(len(y)), query_idx, assume_unique=False)
            return train_idx, np.sort(query_idx)
    splitter = ShuffleSplit(n_splits=1, test_size=query_size, random_state=seed)
    return next(splitter.split(np.zeros((len(y), 1))))


def _make_meta_batch(
    X_context: np.ndarray,
    img_context: torch.Tensor,
    y_context: np.ndarray,
    X_query: np.ndarray,
    img_query: torch.Tensor,
    *,
    classification: bool,
    n_estimators: int,
    pseudo_dim: int,
    norm_methods: list[str],
    feat_shuffle_method: str,
    class_shuffle_method: str,
    outlier_threshold: float,
    seed: int,
    include_image: bool = True,
) -> MetaBatch:
    EnsembleGenerator, Shuffler, TransformToNumerical = _pp_classes()

    yc = y_context.copy()
    y_mean = y_std = None

    raw_table_shape = (len(X_context), X_context.shape[1] if hasattr(X_context, "shape") and len(X_context.shape) > 1 else None)
    table_encoder = TransformToNumerical()
    X_context = np.asarray(table_encoder.fit_transform(X_context), dtype=np.float64)
    X_query = np.asarray(table_encoder.transform(X_query), dtype=np.float64)

    gen = EnsembleGenerator(
        classification=classification,
        n_estimators=n_estimators,
        norm_methods=norm_methods,
        feat_shuffle_method=feat_shuffle_method,
        class_shuffle_method=class_shuffle_method,
        outlier_threshold=outlier_threshold,
        random_state=seed,
    )
    gen.fit(X_context, yc)
    variants = gen.transform(X_query, mode="both")

    table_parts, yctx_parts = [], []
    class_patterns: list[np.ndarray] | None = [] if classification else None
    for norm_method, (Xv, yv) in variants.items():
        table_parts.append(Xv)
        yctx_parts.append(yv)
        if classification:
            assert class_patterns is not None
            for _, pattern in gen.ensemble_configs_[norm_method]:
                class_patterns.append(np.asarray(pattern, dtype=np.int64))

    table = np.concatenate(table_parts, axis=0).astype(np.float32, copy=False)
    yctx = np.concatenate(yctx_parts, axis=0).astype(np.float32, copy=False)
    E, _, H = table.shape

    # Official table preprocessing already includes its own feature shuffles.
    # Add one joint T+V permutation so visual pseudo-columns participate in the
    # same positional augmentation rather than always occupying the last K slots.
    joint = _patterns(
        Shuffler(H + pseudo_dim, method=feat_shuffle_method, random_state=seed + 104729), E
    )

    return MetaBatch(
        table=torch.from_numpy(table),
        image=(torch.cat([img_context, img_query], dim=0).float()
               if include_image else torch.empty(0, dtype=torch.float32)),
        y_context=torch.from_numpy(yctx),
        y_query=torch.empty(E, len(X_query)),
        joint_shuffles=torch.from_numpy(np.stack(joint)).long(),
        class_shuffles=class_patterns,
        context_size=len(X_context),
        raw_table_shape=raw_table_shape,
        y_mean=y_mean,
        y_std=y_std,
    )


def _training_batch(
    X: np.ndarray,
    image: torch.Tensor,
    y: np.ndarray,
    indices: np.ndarray,
    *,
    classification: bool,
    n_estimators: int,
    pseudo_dim: int,
    query_ratio: float,
    norm_methods: list[str],
    feat_shuffle_method: str,
    class_shuffle_method: str,
    outlier_threshold: float,
    seed: int,
) -> MetaBatch:
    Xi, imgi, yi = X[indices], image[indices], y[indices]
    c, q = _split(yi, max(1, int(len(yi) * query_ratio)), seed, classification)

    if classification:
        # Small chunks may contain a subset of global classes, e.g. labels {1, 2}.
        # TabICL's EnsembleGenerator expects classification labels to be local,
        # contiguous indices 0..C_batch-1, so remap within each training meta-batch.
        yc_raw = yi[c].copy()
        yq_raw = yi[q].copy()
        context_classes = np.sort(np.unique(yc_raw))
        query_keep = np.isin(yq_raw, context_classes)
        if not np.all(query_keep):
            c = np.concatenate([c, q[~query_keep]])
            q = q[query_keep]
            yc_raw = yi[c].copy()
            yq_raw = yi[q].copy()
            context_classes = np.sort(np.unique(yc_raw))
        if len(q) == 0:
            counts = {label: int(np.sum(yc_raw == label)) for label in context_classes}
            movable = [label for label, count in counts.items() if count > 1]
            if not movable:
                raise ValueError("Cannot create a classification query split with context covering all query classes")
            label = movable[0]
            move_pos = int(np.flatnonzero(yi[c] == label)[0])
            q = np.asarray([c[move_pos]])
            c = np.delete(c, move_pos)
            yc_raw = yi[c].copy()
            yq_raw = yi[q].copy()
            context_classes = np.sort(np.unique(yc_raw))
        label_map = {label: idx for idx, label in enumerate(context_classes)}
        yc = np.asarray([label_map[label] for label in yc_raw], dtype=np.int64)
        yq = np.asarray([label_map[label] for label in yq_raw], dtype=np.int64)
    else:
        yc, yq = yi[c].copy(), yi[q].copy()

    y_mean = y_std = None
    if not classification:
        y_mean = float(yc.mean())
        y_std = max(float(yc.std()), 1e-8)
        yc = (yc - y_mean) / y_std
        yq = (yq - y_mean) / y_std

    batch = _make_meta_batch(
        Xi[c], imgi[c], yc, Xi[q], imgi[q],
        classification=classification,
        n_estimators=n_estimators,
        pseudo_dim=pseudo_dim,
        norm_methods=norm_methods,
        feat_shuffle_method=feat_shuffle_method,
        class_shuffle_method=class_shuffle_method,
        outlier_threshold=outlier_threshold,
        seed=seed,
    )
    E = batch.table.shape[0]
    if classification:
        assert batch.class_shuffles is not None
        yq_views = [pattern[yq.astype(int)] for pattern in batch.class_shuffles]
        batch.y_query = torch.from_numpy(np.stack(yq_views)).long()
    else:
        batch.y_query = torch.from_numpy(np.stack([yq] * E)).float()
        batch.y_mean, batch.y_std = y_mean, y_std
    return batch


def _eval_batch(
    X_train: np.ndarray,
    img_train: torch.Tensor,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    img_eval: torch.Tensor,
    *,
    classification: bool,
    n_estimators: int,
    pseudo_dim: int,
    norm_methods: list[str],
    feat_shuffle_method: str,
    class_shuffle_method: str,
    outlier_threshold: float,
    seed: int,
    include_image: bool = True,
) -> MetaBatch:
    yc = y_train.copy()
    y_mean = y_std = None
    if not classification:
        y_mean = float(yc.mean())
        y_std = max(float(yc.std()), 1e-8)
        yc = (yc - y_mean) / y_std

    batch = _make_meta_batch(
        X_train, img_train, yc, X_eval, img_eval,
        classification=classification,
        n_estimators=n_estimators,
        pseudo_dim=pseudo_dim,
        norm_methods=norm_methods,
        feat_shuffle_method=feat_shuffle_method,
        class_shuffle_method=class_shuffle_method,
        outlier_threshold=outlier_threshold,
        seed=seed,
        include_image=include_image,
    )
    batch.y_mean, batch.y_std = y_mean, y_std
    return batch


def _chunks(n: int, max_size: int, seed: int) -> list[np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    if n <= max_size:
        return [perm]
    chunks = [perm[i : i + max_size] for i in range(0, n, max_size)]
    return [c for c in chunks if len(c) >= 50] or [perm]


def _to_device(batch: MetaBatch, device: torch.device, *, keep_image_cpu: bool = False) -> MetaBatch:
    batch.table = batch.table.to(device, non_blocking=True)
    if not keep_image_cpu:
        batch.image = batch.image.to(device, non_blocking=True)
    batch.y_context = batch.y_context.to(device, non_blocking=True)
    batch.y_query = batch.y_query.to(device, non_blocking=True)
    batch.joint_shuffles = batch.joint_shuffles.to(device, non_blocking=True)
    return batch


# -----------------------------------------------------------------------------
# Loaders / metrics / eval
# -----------------------------------------------------------------------------


def _load_tabicl(task: str, args) -> nn.Module:
    if task == "classification":
        from tabicl import TabICLClassifier
        est = TabICLClassifier(
            model_path=args.model_path, allow_auto_download=args.allow_auto_download,
            device=args.device, n_estimators=1, batch_size=1, kv_cache=False,
        )
    else:
        from tabicl import TabICLRegressor
        est = TabICLRegressor(
            model_path=args.model_path, allow_auto_download=args.allow_auto_download,
            device=args.device, n_estimators=1, batch_size=1, kv_cache=False,
        )
    est._load_model()
    return est.model_.to(args.device)


def _numeric_splits(splits, task: str):
    X = {s: np.asarray(splits[s].x) for s in ("train", "val", "test")}
    if task == "classification":
        le = LabelEncoder().fit(np.asarray(splits["train"].y).reshape(-1))
        y = {s: le.transform(np.asarray(splits[s].y).reshape(-1)).astype(np.int64) for s in X}
        return X, y, len(le.classes_)
    y = {s: np.asarray(splits[s].y, dtype=np.float32).reshape(-1) for s in X}
    return X, y, None


def _cls_metrics(y, proba):
    pred = proba.argmax(1)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }
    try:
        out["roc_auc"] = float(
            roc_auc_score(y, proba[:, 1]) if proba.shape[1] == 2
            else roc_auc_score(y, proba, multi_class="ovo", average="macro")
        )
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def _reg_metrics(y, pred):
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
    }


def _eval_classification(model, Xtr, Itr, ytr, Xe, Ie, args, n_classes, n_estimators, device, profile=None):
    model.eval()
    pieces = []
    branch_pieces = {"vt": [], "t": []} if model.tabular_shortcut else None
    chunk = len(Xe) if args.evaluation_chunk_size <= 0 else args.evaluation_chunk_size
    eval_started = time.monotonic() if profile is not None else None
    with torch.no_grad():
        # Keep the original 256-row projection boundaries exactly: cache only
        # complete context blocks, then project the context tail together with
        # each query chunk. This avoids both repeated full-context work and
        # batch-shape-dependent floating-point drift in the final block.
        prefix_rows = (len(Itr) // 256) * 256
        context_pseudo = model._pseudo_columns(Itr[:prefix_rows], device) if prefix_rows else None
        if profile is not None:
            torch.cuda.synchronize(device) if device.type == "cuda" else None
            profile["context_precompute_seconds"] = time.monotonic() - eval_started
            profile["query_chunk_seconds"] = []
            profile["query_chunk_rows"] = []
        for start in range(0, len(Xe), chunk):
            chunk_started = time.monotonic() if profile is not None else None
            end = min(len(Xe), start + chunk)
            b = _eval_batch(
                Xtr, Itr, ytr, Xe[start:end], Ie[start:end],
                classification=True, n_estimators=n_estimators, pseudo_dim=args.pseudo_dim,
                norm_methods=args.norm_methods, feat_shuffle_method=args.feat_shuffle_method,
                class_shuffle_method=args.class_shuffle_method, outlier_threshold=args.outlier_threshold,
                seed=args.seed, include_image=False,
            )
            b = _to_device(b, device)
            visual_tail = torch.cat((Itr[prefix_rows:], Ie[start:end]), dim=0)
            tail_pseudo = model._pseudo_columns(visual_tail, device)
            pseudo = torch.cat((context_pseudo, tail_pseudo), dim=0) if context_pseudo is not None else tail_pseudo
            raw_outputs = model.inference_outputs(b, classification=True, precomputed_pseudo=pseudo)
            outputs = (raw_outputs if isinstance(raw_outputs, tuple) else (raw_outputs,))
            assert b.class_shuffles is not None
            probabilities = []
            for output in outputs:
                logits = output[..., :n_classes].float().cpu().numpy()
                avg = np.zeros_like(logits[0])
                for i, pattern in enumerate(b.class_shuffles):
                    avg += logits[i][..., pattern]
                avg /= len(b.class_shuffles)
                avg /= args.softmax_temperature
                avg -= avg.max(-1, keepdims=True)
                avg = np.exp(avg)
                probabilities.append(avg / avg.sum(-1, keepdims=True))
            pieces.append(probabilities[0])
            if branch_pieces is not None:
                branch_pieces["vt"].append(probabilities[1])
                branch_pieces["t"].append(probabilities[2])
            if profile is not None:
                torch.cuda.synchronize(device) if device.type == "cuda" else None
                profile["query_chunk_seconds"].append(time.monotonic() - chunk_started)
                profile["query_chunk_rows"].append(end - start)
    if profile is not None:
        profile["prediction_seconds"] = time.monotonic() - eval_started
        profile["prediction_samples_per_second"] = len(Xe) / profile["prediction_seconds"]
        profile["cold_first_chunk_seconds"] = profile["context_precompute_seconds"] + profile["query_chunk_seconds"][0]
        warm_seconds = sum(profile["query_chunk_seconds"][1:])
        warm_rows = sum(profile["query_chunk_rows"][1:])
        profile["warm_query_seconds"] = warm_seconds
        profile["warm_query_rows"] = warm_rows
        profile["warm_chunks_samples_per_second"] = warm_rows / warm_seconds if warm_seconds else None
    if branch_pieces is not None:
        return {"final": np.concatenate(pieces), **{k: np.concatenate(v) for k, v in branch_pieces.items()}}
    return np.concatenate(pieces)


def _eval_regression(model, Xtr, Itr, ytr, Xe, Ie, args, n_estimators, device, profile=None):
    model.eval()
    pieces = []
    branch_pieces = {"vt": [], "t": []} if model.tabular_shortcut else None
    chunk = len(Xe) if args.evaluation_chunk_size <= 0 else args.evaluation_chunk_size
    eval_started = time.monotonic() if profile is not None else None
    with torch.no_grad():
        prefix_rows = (len(Itr) // 256) * 256
        context_pseudo = model._pseudo_columns(Itr[:prefix_rows], device) if prefix_rows else None
        if profile is not None:
            torch.cuda.synchronize(device) if device.type == "cuda" else None
            profile["context_precompute_seconds"] = time.monotonic() - eval_started
            profile["query_chunk_seconds"] = []
            profile["query_chunk_rows"] = []
        for start in range(0, len(Xe), chunk):
            chunk_started = time.monotonic() if profile is not None else None
            end = min(len(Xe), start + chunk)
            b = _eval_batch(
                Xtr, Itr, ytr, Xe[start:end], Ie[start:end],
                classification=False, n_estimators=n_estimators, pseudo_dim=args.pseudo_dim,
                norm_methods=args.norm_methods, feat_shuffle_method=args.feat_shuffle_method,
                class_shuffle_method=args.class_shuffle_method, outlier_threshold=args.outlier_threshold,
                seed=args.seed, include_image=False,
            )
            b = _to_device(b, device)
            visual_tail = torch.cat((Itr[prefix_rows:], Ie[start:end]), dim=0)
            tail_pseudo = model._pseudo_columns(visual_tail, device)
            pseudo = torch.cat((context_pseudo, tail_pseudo), dim=0) if context_pseudo is not None else tail_pseudo
            raw_outputs = model.inference_outputs(b, classification=False, precomputed_pseudo=pseudo)
            outputs = raw_outputs if isinstance(raw_outputs, tuple) else (raw_outputs,)
            predictions = [value.mean(0).float().cpu().numpy() * b.y_std + b.y_mean for value in outputs]
            if branch_pieces is not None:
                # Fuse the two ensemble predictions in FP32. This is the
                # same linear mixture, but avoids rounding its scalar
                # coefficient in TabICL's internal inference dtype.
                weight = float(model.table_weight.detach().cpu())
                pieces.append((1 - weight) * predictions[1] + weight * predictions[2])
                branch_pieces["vt"].append(predictions[1])
                branch_pieces["t"].append(predictions[2])
            else:
                pieces.append(predictions[0])
            if profile is not None:
                torch.cuda.synchronize(device) if device.type == "cuda" else None
                profile["query_chunk_seconds"].append(time.monotonic() - chunk_started)
                profile["query_chunk_rows"].append(end - start)
    if profile is not None:
        profile["prediction_seconds"] = time.monotonic() - eval_started
        profile["prediction_samples_per_second"] = len(Xe) / profile["prediction_seconds"]
        profile["cold_first_chunk_seconds"] = profile["context_precompute_seconds"] + profile["query_chunk_seconds"][0]
        warm_seconds = sum(profile["query_chunk_seconds"][1:])
        warm_rows = sum(profile["query_chunk_rows"][1:])
        profile["warm_query_seconds"] = warm_seconds
        profile["warm_query_rows"] = warm_rows
        profile["warm_chunks_samples_per_second"] = warm_rows / warm_seconds if warm_seconds else None
    if branch_pieces is not None:
        return {"final": np.concatenate(pieces), **{k: np.concatenate(v) for k, v in branch_pieces.items()}}
    return np.concatenate(pieces)


def _scheduler(optimizer, total_steps: int, warmup: float):
    cfg = SimpleNamespace(
        max_steps=max(1, total_steps),
        warmup_proportion=warmup,
        warmup_steps=0,
        scheduler="cosine_warmup",
    )
    return tabicl_get_scheduler(cfg, optimizer)


def _select_rows(y, maximum, seed, classification):
    if maximum <= 0 or maximum >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    if not classification:
        return np.sort(rng.choice(len(y), maximum, replace=False))
    labels = np.unique(y)
    if maximum < len(labels):
        raise ValueError("max_train_context is smaller than the number of classes")
    chosen = [rng.choice(np.flatnonzero(y == c)) for c in labels]
    rest = np.setdiff1d(np.arange(len(y)), chosen)
    chosen += rng.choice(rest, maximum - len(chosen), replace=False).tolist()
    return np.sort(np.asarray(chosen))


def _grad_norm(params):
    return math.sqrt(sum(float(p.grad.detach().float().norm().cpu()) ** 2 for p in params if p.grad is not None))


def _grads_finite(params) -> bool:
    seen = False
    for p in params:
        if p.grad is None:
            continue
        seen = True
        if not torch.isfinite(p.grad.detach()).all().item():
            return False
    return seen


def _phase2_regression_loss(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    q = out.shape[-1]
    alpha = torch.linspace(0.0, 1.0, q + 2, device=out.device, dtype=out.dtype)[1:-1]
    return tabicl_pinball_loss(out, target, alpha)


def _count_params(params) -> int:
    return sum(p.numel() for p in params)


def _optimizer_param_stats(optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    return len(ids), len(ids) - len(set(ids))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_phase1_init_checkpoint(args, state: dict, checkpoint_path: Path) -> None:
    if args.phase != 2:
        return
    primary_metric, _ = get_primary_protocol(get_dataset_spec(args.dataset).task)
    expected = {
        "phase": 1,
        "fusion_mode": "pseudo_mlp",
        "pseudo_dim": args.pseudo_dim,
        "dataset": args.dataset,
        "seed": args.seed,
        "primary_protocol_version": PRIMARY_PROTOCOL_VERSION,
        "primary_metric": primary_metric,
    }
    for key, want in expected.items():
        got = state.get(key)
        if got is not None and got != want:
            raise ValueError(f"Phase2 init checkpoint {key}={got!r} does not match required {want!r}")
    if args.image_encoder == "clip_vitl14_multilayer" and state.get("image_encoder") != args.image_encoder:
        raise ValueError("Phase2 multi-layer CLIP requires a Phase1 multi-layer CLIP checkpoint")

    metrics_path = checkpoint_path.parent / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Cannot verify Phase2 init checkpoint without sibling metrics.json: {metrics_path}"
        )
    meta = json.loads(metrics_path.read_text())
    for key, want in expected.items():
        got = meta.get(key)
        if got != want:
            raise ValueError(
                f"Phase2 init metrics {metrics_path} has {key}={got!r}, expected {want!r}"
            )
    recorded_checkpoint = meta.get("checkpoint")
    if not recorded_checkpoint:
        raise ValueError(f"Phase1 metrics {metrics_path} is missing checkpoint provenance")
    if Path(recorded_checkpoint).expanduser().resolve() != checkpoint_path:
        raise ValueError(
            f"Phase1 metrics checkpoint={recorded_checkpoint!r} does not match init checkpoint={str(checkpoint_path)!r}"
        )


# -----------------------------------------------------------------------------
# CLI / train
# -----------------------------------------------------------------------------


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=("E1", "E2", "E3", "E4"), default="E3")
    p.add_argument("--vision", choices=("clip", "dinov2", "openclip"), default="clip")
    p.add_argument("--modality", choices=("full", "without-image", "without-table"), default="full")
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--embedding-root", type=Path)
    p.add_argument("--image-encoder", choices=("clip_vitl14_multilayer",), default="clip_vitl14_multilayer")
    p.add_argument("--image-model-id")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embedding-batch-size", type=int, default=16)

    p.add_argument("--fusion-mode", choices=("pseudo_mlp",), required=True)
    p.add_argument("--phase", type=int, choices=(1, 2), default=1)
    p.add_argument("--phase2-mode", choices=("backbone_only", "joint"))
    p.add_argument("--init-checkpoint", type=Path)
    p.add_argument("--pseudo-dim", type=int, choices=(4, 8, 16, 32, 64, 128), default=8)
    p.add_argument("--tabular-shortcut", action="store_true", help="Frozen independent table-only TabICL with a trainable dataset-level prediction mixture")
    p.add_argument("--shortcut-learning-rate", type=float, default=1e-3)
    p.add_argument("--shortcut-loss-divisor", type=float, choices=(1.0, 2.0), default=1.0)
    p.add_argument("--mgm-heads", type=int, default=8)
    p.add_argument("--cap-attention-heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--projector-learning-rate", type=float)
    p.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    p.add_argument("--projector-weight-decay", type=float, default=1e-4)
    p.add_argument("--backbone-weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-proportion", type=float, default=0.1)
    p.add_argument("--no-scheduler", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--amp-phase1", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--debug-gradient-steps", type=int, default=1)

    p.add_argument("--n-estimators-finetune", type=int, default=2)
    p.add_argument("--n-estimators-validation", type=int, default=2)
    p.add_argument("--n-estimators-inference", type=int, default=8)
    p.add_argument("--max-data-size", type=int, default=10_000)
    p.add_argument("--query-ratio", type=float, default=0.2)
    p.add_argument("--norm-methods", nargs="+", default=["none", "power"])
    p.add_argument("--feat-shuffle-method", choices=("none", "shift", "random", "latin"), default="latin")
    p.add_argument("--class-shuffle-method", choices=("none", "shift", "random", "latin"), default="shift")
    p.add_argument("--outlier-threshold", type=float, default=4.0)
    p.add_argument("--softmax-temperature", type=float, default=0.9)

    p.add_argument("--max-train-context", type=int, default=0)
    p.add_argument("--evaluation-chunk-size", type=int, default=512)
    p.add_argument("--time-limit", type=int, default=0)
    p.add_argument("--model-path", type=Path)
    p.add_argument("--allow-auto-download", action="store_true")
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--test-only", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    return p.parse_args()


def main():
    process_started = time.monotonic()
    args = _args()
    if args.modality != "full" and not args.test_only:
        raise ValueError("Modality masking requires --test-only.")
    if args.variant != "E3" and (args.phase != 1 or args.tabular_shortcut):
        raise ValueError("Paper ablations require a frozen TabICLv2 predictor.")
    multilayer_clip = args.image_encoder == "clip_vitl14_multilayer"
    if args.vision != "clip" and args.variant != "E3":
        raise ValueError("Alternative vision encoders require the standard patch count.")
    if args.test_only and args.skip_test:
        raise ValueError("--test-only and --skip-test are mutually exclusive")
    if args.test_only and args.checkpoint is None:
        raise ValueError("--test-only requires --checkpoint")
    if args.tabular_shortcut and (args.phase != 1 or args.fusion_mode != "pseudo_mlp"):
        raise ValueError("--tabular-shortcut requires Phase 1 pseudo_mlp")
    if not args.tabular_shortcut and args.shortcut_loss_divisor != 1.0:
        raise ValueError("--shortcut-loss-divisor requires --tabular-shortcut")
    if args.fusion_mode == "pseudo_mmpfn" and args.pseudo_dim in {64, 128}:
        raise ValueError("pseudo_dim=64/128 is only enabled for fusion_mode=pseudo_mlp")
    if multilayer_clip and args.fusion_mode != "pseudo_mlp":
        raise ValueError("clip_vitl14_multilayer requires --fusion-mode pseudo_mlp")
    if args.phase == 2:
        if args.phase2_mode is None:
            raise ValueError("--phase 2 requires --phase2-mode {backbone_only,joint}")
        if args.fusion_mode != "pseudo_mlp":
            raise ValueError("Phase 2 is currently implemented only for fusion_mode=pseudo_mlp")
    elif args.phase2_mode is not None:
        raise ValueError("--phase2-mode is only valid with --phase 2")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    spec = get_dataset_spec(args.dataset)
    if multilayer_clip and spec.secondary_modality != "image":
        raise ValueError("clip_vitl14_multilayer requires an image dataset")
    primary_metric, higher_is_better = get_primary_protocol(spec.task)
    data_started = time.monotonic()
    splits = load_benchmark_splits(spec, args.data_root)
    X, y, n_classes = _numeric_splits(splits, spec.task)
    data_preprocess_seconds = time.monotonic() - data_started

    embedding_root = args.embedding_root or (args.output_dir / "embeddings" / args.dataset)
    embedding_started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    images = {}
    for s, ds in splits.items():
        if args.vision != "clip" or args.variant == "E4":
            from .vision import extract_tokens
            raw = extract_tokens(
                ds, Path(embedding_root) / f"{s}_{args.image_encoder}_raw.pt",
                args.image_model_id, args.vision, args.device, args.embedding_batch_size,
                patches_per_layer=256 if args.variant == "E4" else 64,
            ).float()
        else:
            raw = ds.get_embeddings(
                None, Path(embedding_root) / f"{s}_{args.image_encoder}_raw.pt",
                args.embedding_batch_size, args.device,
                image_encoder="clip_vitl14" if multilayer_clip else args.image_encoder,
                image_model_id=args.image_model_id,
                token_type="patch" if multilayer_clip else "cls",
                clip_spatial_patch_layers=(6, 12, 18, 24) if multilayer_clip else None,
            ).detach().cpu().float()
        images[s] = raw.unsqueeze(1) if multilayer_clip else raw
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        feature_peak_cuda_allocated_bytes = torch.cuda.max_memory_allocated(device)
        feature_peak_cuda_reserved_bytes = torch.cuda.max_memory_reserved(device)
    else:
        feature_peak_cuda_allocated_bytes = None
        feature_peak_cuda_reserved_bytes = None
    feature_extraction_or_cache_load_seconds = time.monotonic() - embedding_started
    embedding_load_seconds = time.monotonic() - process_started

    if multilayer_clip:
        # Loading CLIP on a cache miss consumes RNG state, whereas a cache hit
        # does not. Keep downstream projector initialization and training tied
        # to the requested seed rather than to the embedding cache state.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    idx = _select_rows(y["train"], args.max_train_context, args.seed, spec.task == "classification")
    # Advanced indexing makes a full copy of the (large) CLIP tensor.  With
    # unlimited context, idx is the identity, so keep the original storage.
    if len(idx) != len(y["train"]):
        X["train"], y["train"], images["train"] = X["train"][idx], y["train"][idx], images["train"][idx]

    model = TabICLPseudoColumnModel(
        _load_tabicl(spec.task, args), int(images["train"].shape[-1]), args.fusion_mode, args.pseudo_dim,
        mgm_heads=args.mgm_heads, cap_attention_heads=args.cap_attention_heads, dropout=args.dropout,
        multilayer_clip=multilayer_clip,
        tabular_shortcut=args.tabular_shortcut,
    ).to(device)
    phase1_init_checkpoint = None
    phase1_init_sha256 = None
    if args.phase == 2 and args.init_checkpoint is None and not args.test_only:
        raise ValueError("Phase 2 requires --init-checkpoint from the selected Phase-1 run")
    if args.init_checkpoint:
        phase1_init_checkpoint = args.init_checkpoint.expanduser().resolve()
        state = torch.load(phase1_init_checkpoint, map_location="cpu", weights_only=False)
        _validate_phase1_init_checkpoint(args, state, phase1_init_checkpoint)
        phase1_init_sha256 = _sha256_file(phase1_init_checkpoint)
        model.load_state_dict(state["model_state_dict"], strict=True)
    model.modality = args.modality
    fusion = model.visual_multilayer_fusion
    if fusion is not None:
        fusion.variant = args.variant
        if args.variant == "E4":
            fusion.patches_per_layer = 256
    model.configure_phase(args.phase, args.phase2_mode)

    fusion_root = Path(args.output_dir) / args.dataset / (args.data_root.name if args.data_root.name.startswith("fold_") else "fold_0") / args.fusion_mode
    if multilayer_clip:
        fusion_root = fusion_root / args.image_encoder
    if args.tabular_shortcut:
        fusion_root = fusion_root / f"tabular_shortcut_lossdiv{int(args.shortcut_loss_divisor)}"
    run_root = fusion_root / f"k{args.pseudo_dim}" / ("phase1" if args.phase == 1 else "phase2")
    if args.phase == 2:
        run_root = run_root / args.phase2_mode
    run_root = run_root / f"seed_{args.seed}"
    run_root.mkdir(parents=True, exist_ok=True)

    if args.test_only:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        expected_ckpt = {
            "fusion_mode": args.fusion_mode,
            "pseudo_dim": args.pseudo_dim,
            "phase": args.phase,
            "dataset": args.dataset,
            "seed": args.seed,
            "primary_protocol_version": PRIMARY_PROTOCOL_VERSION,
            "primary_metric": primary_metric,
        }
        if args.tabular_shortcut:
            expected_ckpt["tabular_shortcut"] = True
            expected_ckpt["shortcut_loss_divisor"] = args.shortcut_loss_divisor
        if multilayer_clip:
            expected_ckpt["image_encoder"] = args.image_encoder
        if args.phase == 2:
            expected_ckpt["phase2_mode"] = args.phase2_mode
        for key, want in expected_ckpt.items():
            got = ckpt.get(key)
            if got is None:
                raise ValueError(f"Checkpoint is missing required provenance field {key!r}: {checkpoint_path}")
            if key in {"pseudo_dim", "phase", "seed"}:
                got = int(got)
            if got != want:
                raise ValueError(
                    f"Checkpoint {key}={got!r} does not match requested {want!r}"
                )
        if args.phase == 2 and args.phase2_mode == "backbone_only":
            if ckpt.get("projector_trainable") is not False or ckpt.get("tabicl_trainable") is not True:
                raise ValueError(
                    "backbone_only checkpoint must record projector_trainable=False and tabicl_trainable=True"
                )

        test_phase1_init_checkpoint = ckpt.get("phase1_init_checkpoint")
        test_phase1_init_sha256 = ckpt.get("phase1_init_sha256")
        if args.phase == 2:
            if not test_phase1_init_checkpoint:
                raise ValueError(f"Phase2 checkpoint is missing phase1_init_checkpoint: {checkpoint_path}")
            init_path = Path(test_phase1_init_checkpoint).expanduser().resolve()
            if not init_path.is_file():
                raise FileNotFoundError(f"Phase1 init checkpoint referenced by Phase2 checkpoint not found: {init_path}")
            computed_init_sha256 = _sha256_file(init_path)
            if test_phase1_init_sha256 is not None and test_phase1_init_sha256 != computed_init_sha256:
                raise ValueError(
                    f"Phase1 init SHA256 mismatch: checkpoint recorded {test_phase1_init_sha256}, "
                    f"current file is {computed_init_sha256}"
                )
            test_phase1_init_sha256 = computed_init_sha256

        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        print(
            f"checkpoint loaded: {checkpoint_path} "
            f"dataset={args.dataset} method={args.fusion_mode} K={args.pseudo_dim} "
            f"best_epoch={ckpt.get('epoch', 'unknown')}",
            flush=True,
        )

        inference_profile = {}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        if spec.task == "classification":
            pred = _eval_classification(
                model,
                X["train"],
                images["train"],
                y["train"],
                X["test"],
                images["test"],
                args,
                n_classes,
                args.n_estimators_inference,
                device,
                profile=inference_profile,
            )
            if args.tabular_shortcut:
                branch_test_metrics = {name: _cls_metrics(y["test"], values) for name, values in pred.items()}
                test_metrics = branch_test_metrics["final"]
            else:
                test_metrics = _cls_metrics(y["test"], pred)
        else:
            pred = _eval_regression(
                model,
                X["train"],
                images["train"],
                y["train"],
                X["test"],
                images["test"],
                args,
                args.n_estimators_inference,
                device,
                profile=inference_profile,
            )
            if args.tabular_shortcut:
                branch_test_metrics = {name: _reg_metrics(y["test"], values) for name, values in pred.items()}
                test_metrics = branch_test_metrics["final"]
            else:
                test_metrics = _reg_metrics(y["test"], pred)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            inference_profile["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            inference_profile["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        inference_profile["embedding_load_seconds"] = embedding_load_seconds
        inference_profile["data_preprocess_seconds"] = data_preprocess_seconds
        inference_profile["feature_extraction_or_cache_load_seconds"] = feature_extraction_or_cache_load_seconds
        inference_profile["feature_peak_cuda_allocated_bytes"] = feature_peak_cuda_allocated_bytes
        inference_profile["feature_peak_cuda_reserved_bytes"] = feature_peak_cuda_reserved_bytes
        inference_profile["total_seconds"] = time.monotonic() - process_started
        inference_profile["setup_seconds"] = max(
            0.0,
            inference_profile["total_seconds"] - embedding_load_seconds - inference_profile["prediction_seconds"],
        )
        try:
            import resource
            inference_profile["peak_cpu_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except ImportError:
            inference_profile["peak_cpu_rss_kib"] = None
        inference_profile["checkpoint_bytes"] = checkpoint_path.stat().st_size
        inference_profile["cache_bytes"] = {
            split: (Path(embedding_root) / f"{split}_{args.image_encoder}_raw.pt").stat().st_size
            for split in ("train", "val", "test")
        }

        if primary_metric not in test_metrics:
            raise RuntimeError(
                f"Primary metric {primary_metric!r} missing from test metrics; "
                f"available={sorted(test_metrics)}"
            )
        result = {
            "dataset": args.dataset,
            "task": spec.task,
            "fusion_mode": args.fusion_mode,
            "phase": args.phase,
            "phase2_mode": args.phase2_mode,
            "pseudo_dim": args.pseudo_dim,
            "backbone": "TabICLv2",
            "image_encoder": args.image_encoder,
            "seed": args.seed,
            "checkpoint": str(checkpoint_path),
            "phase1_init_checkpoint": test_phase1_init_checkpoint,
            "phase1_init_sha256": test_phase1_init_sha256,
            "projector_trainable": ckpt.get("projector_trainable"),
            "tabicl_trainable": ckpt.get("tabicl_trainable"),
            "projector_learning_rate": ckpt.get("projector_learning_rate"),
            "backbone_learning_rate": ckpt.get("backbone_learning_rate"),
            "best_epoch": ckpt.get("epoch"),
            "best_validation_value": (ckpt.get("validation_metrics") or {}).get(primary_metric),
            "fusion_weights": model.visual_multilayer_fusion.weight_snapshot() if multilayer_clip else None,
            "validation_metrics": ckpt.get("validation_metrics"),
            "test_evaluated": True,
            "tabular_shortcut": args.tabular_shortcut,
            "shortcut_loss_divisor": args.shortcut_loss_divisor if args.tabular_shortcut else None,
            "table_weight": float(model.table_weight.detach().cpu()) if args.tabular_shortcut else None,
            "branch_test_metrics": branch_test_metrics if args.tabular_shortcut else None,
            "primary_protocol_version": PRIMARY_PROTOCOL_VERSION,
            "primary_metric": primary_metric,
            "primary_value": test_metrics[primary_metric],
            "n_train": int(len(y["train"])),
            "n_val": int(len(y["val"])),
            "n_test": int(len(y["test"])),
            "resource_usage": inference_profile,
            **test_metrics,
        }
        test_path = run_root / ("test_metrics.json" if args.modality == "full" else f"test_{args.modality}.json")
        test_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        print(f"test metrics saved: {test_path}", flush=True)
        return

    projector_lr = args.projector_learning_rate
    if projector_lr is None:
        if args.phase == 2 and args.phase2_mode == "joint":
            projector_lr = 1e-4
        else:
            projector_lr = 1e-3 if args.fusion_mode == "pseudo_mlp" else 3e-4
    if args.phase == 2 and args.phase2_mode == "backbone_only":
        projector_lr = 0.0
    proj = [p for p in model.pseudo_parameters() if p.requires_grad]
    backbone = [p for p in model.tabicl.parameters() if p.requires_grad]
    if args.phase == 1:
        if args.tabular_shortcut:
            optimizer = torch.optim.AdamW([
                {"params": proj, "lr": projector_lr, "weight_decay": args.projector_weight_decay, "name": "visual"},
                {"params": [model.table_weight], "lr": args.shortcut_learning_rate, "weight_decay": 0.0, "name": "table_weight"},
            ])
        elif multilayer_clip:
            fusion = model.visual_multilayer_fusion
            logits = [p for p in (fusion.layer_logits, fusion.branch_logits) if p.requires_grad]
            logit_ids = {id(parameter) for parameter in logits}
            optimizer = torch.optim.AdamW(
                [
                    {"params": [p for p in proj if id(p) not in logit_ids], "lr": projector_lr},
                    *([{"params": logits, "lr": 1e-2}] if logits else []),
                ],
                weight_decay=args.projector_weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(proj, lr=projector_lr, weight_decay=args.projector_weight_decay)
    else:
        param_groups = [
            {"params": backbone, "lr": args.backbone_learning_rate, "weight_decay": args.backbone_weight_decay, "name": "tabicl"}
        ]
        if args.phase2_mode == "joint":
            param_groups.append({
                "params": proj,
                "lr": projector_lr,
                "weight_decay": args.projector_weight_decay,
                "name": "pseudo_projector",
            })
        optimizer = torch.optim.AdamW(param_groups)
    optimizer_param_count, duplicate_optimizer_params = _optimizer_param_stats(optimizer)
    if duplicate_optimizer_params:
        raise RuntimeError(f"Optimizer contains {duplicate_optimizer_params} duplicate parameters")
    projector_trainable_params = _count_params(proj)
    tabicl_trainable_params = _count_params(backbone)

    chunks_per_epoch = max(1, len(_chunks(len(y["train"]), args.max_data_size, args.seed)))
    scheduler = None if args.no_scheduler else _scheduler(
        optimizer, args.epochs * chunks_per_epoch, args.warmup_proportion
    )
    use_amp = device.type == "cuda" and not args.no_amp and (args.phase == 2 or args.amp_phase1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        initial_cuda_allocated_bytes = torch.cuda.memory_allocated(device)
        initial_cuda_reserved_bytes = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        initial_cuda_allocated_bytes = None
        initial_cuda_reserved_bytes = None
    setup_seconds = time.monotonic() - process_started - embedding_load_seconds
    train_seconds = 0.0
    validation_seconds = 0.0

    best_path = run_root / "best_pseudo_columns.pt"
    best = -float("inf") if higher_is_better else float("inf")
    best_epoch, patience = -1, 0
    history = []
    started = time.monotonic()

    def run_validation():
        if spec.task == "classification":
            val_pred = _eval_classification(
                model, X["train"], images["train"], y["train"], X["val"], images["val"],
                args, n_classes, args.n_estimators_validation, device,
            )
            if args.tabular_shortcut:
                branch_metrics = {name: _cls_metrics(y["val"], values) for name, values in val_pred.items()}
                metrics = {**branch_metrics["final"], **{
                    f"{name}_{key}": value for name in ("vt", "t") for key, value in branch_metrics[name].items()
                }}
            else:
                metrics = _cls_metrics(y["val"], val_pred)
        else:
            val_pred = _eval_regression(
                model, X["train"], images["train"], y["train"], X["val"], images["val"],
                args, args.n_estimators_validation, device,
            )
            if args.tabular_shortcut:
                branch_metrics = {name: _reg_metrics(y["val"], values) for name, values in val_pred.items()}
                metrics = {**branch_metrics["final"], **{
                    f"{name}_{key}": value for name in ("vt", "t") for key, value in branch_metrics[name].items()
                }}
            else:
                metrics = _reg_metrics(y["val"], val_pred)
        if primary_metric not in metrics:
            raise RuntimeError(
                f"Primary metric {primary_metric!r} missing from validation metrics; "
                f"available={sorted(metrics)}"
            )
        return metrics, metrics[primary_metric]

    if args.phase == 2:
        val_metrics, best = run_validation()
        torch.save({
            "model_state_dict": model.state_dict(), "epoch": -1, "phase": args.phase,
            "image_encoder": args.image_encoder,
            "phase2_mode": args.phase2_mode,
            "fusion_mode": args.fusion_mode, "pseudo_dim": args.pseudo_dim,
            "dataset": args.dataset, "seed": args.seed,
            "primary_protocol_version": PRIMARY_PROTOCOL_VERSION,
            "primary_metric": primary_metric,
            "phase1_init_checkpoint": str(phase1_init_checkpoint),
            "phase1_init_sha256": phase1_init_sha256,
            "projector_trainable": projector_trainable_params > 0,
            "tabicl_trainable": tabicl_trainable_params > 0,
            "projector_learning_rate": projector_lr,
            "backbone_learning_rate": args.backbone_learning_rate,
            "validation_metrics": val_metrics,
            "fusion_weights": model.visual_multilayer_fusion.weight_snapshot() if multilayer_clip else None,
            "tabular_shortcut": args.tabular_shortcut,
            "shortcut_loss_divisor": args.shortcut_loss_divisor if args.tabular_shortcut else None,
            "table_weight": float(model.table_weight.detach().cpu()) if args.tabular_shortcut else None,
        }, best_path)
        history.append({
            "epoch": -1, "training_loss": None,
            **{f"validation_{k}": v for k, v in val_metrics.items()},
            **(model.visual_multilayer_fusion.weight_snapshot() if multilayer_clip else {}),
        })
        best_epoch = -1
        print(f"baseline val={best:.6f} best={best_epoch}", flush=True)

    for epoch in range(args.epochs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_started = time.monotonic()
        model.train(True)
        losses = []
        branch_losses = {"final": [], "vt": [], "t": []} if args.tabular_shortcut else None
        first = True
        for ci, chunk_idx in enumerate(_chunks(len(y["train"]), args.max_data_size, args.seed + epoch)):
            b = _training_batch(
                X["train"], images["train"], y["train"], chunk_idx,
                classification=spec.task == "classification",
                n_estimators=args.n_estimators_finetune, pseudo_dim=args.pseudo_dim,
                query_ratio=args.query_ratio, norm_methods=args.norm_methods,
                feat_shuffle_method=args.feat_shuffle_method, class_shuffle_method=args.class_shuffle_method,
                outlier_threshold=args.outlier_threshold, seed=args.seed + epoch + ci * 7919,
            )
            b = _to_device(b, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model.training_outputs(b, classification=spec.task == "classification")
                if spec.task == "classification":
                    batch_n_classes = int(max(b.y_context.max(), b.y_query.max()).item()) + 1
                    if args.tabular_shortcut:
                        final_out, vt_out, t_out = out
                        target = b.y_query.reshape(-1)
                        final_loss = torch.nn.functional.cross_entropy(
                            final_out[..., :batch_n_classes].reshape(-1, batch_n_classes).float(), target
                        )
                        vt_loss = torch.nn.functional.cross_entropy(
                            vt_out[..., :batch_n_classes].reshape(-1, batch_n_classes).float(), target
                        )
                        with torch.no_grad():
                            t_loss = torch.nn.functional.cross_entropy(
                                t_out[..., :batch_n_classes].reshape(-1, batch_n_classes).float(), target
                            )
                        loss = (final_loss + vt_loss) / args.shortcut_loss_divisor
                        for name, value in (("final", final_loss), ("vt", vt_loss), ("t", t_loss)):
                            branch_losses[name].append(float(value.detach().cpu()))
                    else:
                        out = out[..., :batch_n_classes]
                        loss = torch.nn.functional.cross_entropy(
                            out.reshape(-1, batch_n_classes).float(), b.y_query.reshape(-1)
                        )
                else:
                    if args.phase == 2:
                        loss = _phase2_regression_loss(out, b.y_query)
                    elif args.tabular_shortcut:
                        final_out, vt_out, t_out = out
                        target = b.y_query.float()
                        final_loss = torch.nn.functional.mse_loss(final_out.float(), target)
                        vt_loss = torch.nn.functional.mse_loss(vt_out.float(), target)
                        with torch.no_grad():
                            t_loss = torch.nn.functional.mse_loss(t_out.float(), target)
                        loss = (final_loss + vt_loss) / args.shortcut_loss_divisor
                        for name, value in (("final", final_loss), ("vt", vt_loss), ("t", t_loss)):
                            branch_losses[name].append(float(value.detach().cpu()))
                    else:
                        pred = model.tabicl.quantile_dist(out).quantiles.mean(-1)
                        loss = torch.nn.functional.mse_loss(pred.float(), b.y_query.float())
            scale_before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            projector_unscaled_grad_norm = _grad_norm(iter(proj))
            tabicl_unscaled_grad_norm = _grad_norm(iter(model.tabicl.parameters()))
            projector_grad_finite = _grads_finite(iter(proj)) if proj else True
            tabicl_grad_finite = _grads_finite(iter(backbone)) if backbone else True
            optimizer_grads_finite = _grads_finite(p for group in optimizer.param_groups for p in group["params"])
            if first:
                tabicl_grads_none = all(p.grad is None for p in model.tabicl.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                tabicl_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
                projector_lr_active = 0.0
                if args.phase == 1:
                    projector_lr_active = optimizer.param_groups[0]["lr"]
                elif args.phase2_mode == "joint" and len(optimizer.param_groups) > 1:
                    projector_lr_active = optimizer.param_groups[1]["lr"]
                parts = [
                    f"phase={args.phase}",
                    f"phase2_mode={args.phase2_mode}",
                    f"selected_k={args.pseudo_dim}",
                    f"phase1_checkpoint={phase1_init_checkpoint}",
                    f"total_trainable_params={trainable_params}",
                    f"projector_trainable_params={projector_trainable_params}",
                    f"tabicl_trainable_params={tabicl_trainable_params}",
                    f"grad_scaler_scale={scale_before}",
                    f"projector_unscaled_grad_norm={projector_unscaled_grad_norm:.4f}",
                    f"tabicl_unscaled_grad_norm={tabicl_unscaled_grad_norm:.4f}",
                    f"projector_grad_finite={projector_grad_finite}",
                    f"tabicl_grad_finite={tabicl_grad_finite}",
                    f"tabicl_all_grad_none={tabicl_grads_none}",
                    f"projector_training={any(m.training for m in (model.visual_pseudo_mlp, model.visual_multilayer_fusion, model.visual_mgm, model.visual_cap, model.visual_scalar_head) if m is not None)}",
                    f"tabicl_training={model.tabicl.training}",
                    f"projector_lr={projector_lr_active}",
                    f"tabicl_lr={tabicl_lr}",
                    f"projector_base_lr={projector_lr}",
                    f"tabicl_base_lr={args.backbone_learning_rate if args.phase == 2 else 0.0}",
                    f"optimizer_parameter_count={optimizer_param_count}",
                    f"duplicate_optimizer_parameters={duplicate_optimizer_params}",
                ]
                if model.fusion_mode == "pseudo_mmpfn":
                    parts.extend([
                        f"mgm_grad_norm={_grad_norm(model.visual_mgm.parameters()):.4f}",
                        f"cap_grad_norm={_grad_norm(model.visual_cap.parameters()):.4f}",
                        f"scalar_head_grad_norm={_grad_norm(model.visual_scalar_head.parameters()):.4f}",
                    ])
                print("backward debug: " + " ".join(parts), flush=True)
                first = False
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]], args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if args.tabular_shortcut:
                with torch.no_grad():
                    model.table_weight.clamp_(0.0, 1.0)
            scale_after = float(scaler.get_scale())
            optimizer_step_executed = (not use_amp) or (optimizer_grads_finite and scale_after >= scale_before)
            global_step_index = epoch * chunks_per_epoch + ci
            if global_step_index < args.debug_gradient_steps:
                print(
                    "gradient debug: "
                    f"step={global_step_index} "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"grad_scaler_scale={scale_before} "
                    f"projector_unscaled_grad_norm={projector_unscaled_grad_norm:.4f} "
                    f"tabicl_unscaled_grad_norm={tabicl_unscaled_grad_norm:.4f} "
                    f"projector_grad_finite={projector_grad_finite} "
                    f"tabicl_grad_finite={tabicl_grad_finite} "
                    f"optimizer_step_executed={optimizer_step_executed}",
                    flush=True,
                )
            if scheduler and optimizer_step_executed:
                scheduler.step()
            losses.append(float(loss.detach().cpu()))

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_train_seconds = time.monotonic() - epoch_started
        train_seconds += epoch_train_seconds
        val_started = time.monotonic()
        val_metrics, current = run_validation()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_validation_seconds = time.monotonic() - val_started
        validation_seconds += epoch_validation_seconds

        improved = current > best + args.min_delta if higher_is_better else current < best - args.min_delta
        should_save = improved or (args.phase == 1 and best_epoch < 0)
        if should_save:
            best, best_epoch, patience = current, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch, "phase": args.phase,
                "image_encoder": args.image_encoder,
                "phase2_mode": args.phase2_mode,
                "fusion_mode": args.fusion_mode, "pseudo_dim": args.pseudo_dim,
                "dataset": args.dataset, "seed": args.seed,
                "primary_protocol_version": PRIMARY_PROTOCOL_VERSION,
                "primary_metric": primary_metric,
                "phase1_init_checkpoint": str(phase1_init_checkpoint) if phase1_init_checkpoint is not None else None,
                "phase1_init_sha256": phase1_init_sha256,
                "projector_trainable": projector_trainable_params > 0,
                "tabicl_trainable": tabicl_trainable_params > 0,
                "projector_learning_rate": projector_lr,
                "backbone_learning_rate": args.backbone_learning_rate if args.phase == 2 else 0.0,
                "validation_metrics": val_metrics,
                "fusion_weights": model.visual_multilayer_fusion.weight_snapshot() if multilayer_clip else None,
                "tabular_shortcut": args.tabular_shortcut,
                "shortcut_loss_divisor": args.shortcut_loss_divisor if args.tabular_shortcut else None,
                "table_weight": float(model.table_weight.detach().cpu()) if args.tabular_shortcut else None,
            }, best_path)
        else:
            patience += 1
        mean_loss = float(np.mean(losses))
        history.append({
            "epoch": epoch, "training_loss": mean_loss,
            **({"table_weight": float(model.table_weight.detach().cpu()),
                **{f"training_loss_{name}": float(np.mean(values)) for name, values in branch_losses.items()}}
               if args.tabular_shortcut else {}),
            "train_seconds": epoch_train_seconds,
            "validation_seconds": epoch_validation_seconds,
            **{f"validation_{k}": v for k, v in val_metrics.items()},
            **(model.visual_multilayer_fusion.weight_snapshot() if multilayer_clip else {}),
        })
        print(f"epoch={epoch:02d} loss={mean_loss:.6f} val={current:.6f} best={best_epoch} patience={patience}/{args.patience}", flush=True)
        if args.time_limit and time.monotonic() - started > args.time_limit: break
        if args.patience > 0 and patience >= args.patience: break

    ckpt = torch.load(best_path, map_location=device if not args.skip_test else "cpu", weights_only=False)
    test_metrics = None
    if not args.skip_test:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if spec.task == "classification":
            pred = _eval_classification(
                model, X["train"], images["train"], y["train"], X["test"], images["test"],
                args, n_classes, args.n_estimators_inference, device,
            )
            if args.tabular_shortcut:
                branch_test_metrics = {name: _cls_metrics(y["test"], values) for name, values in pred.items()}
                test_metrics = branch_test_metrics["final"]
            else:
                test_metrics = _cls_metrics(y["test"], pred)
        else:
            pred = _eval_regression(
                model, X["train"], images["train"], y["train"], X["test"], images["test"],
                args, args.n_estimators_inference, device,
            )
            if args.tabular_shortcut:
                branch_test_metrics = {name: _reg_metrics(y["test"], values) for name, values in pred.items()}
                test_metrics = branch_test_metrics["final"]
            else:
                test_metrics = _reg_metrics(y["test"], pred)

    if test_metrics is not None and primary_metric not in test_metrics:
        raise RuntimeError(
            f"Primary metric {primary_metric!r} missing from test metrics; "
            f"available={sorted(test_metrics)}"
        )

    result = {
        "dataset": args.dataset, "task": spec.task, "fusion_mode": args.fusion_mode,
        "phase": args.phase, "phase2_mode": args.phase2_mode,
        "pseudo_dim": args.pseudo_dim, "backbone": "TabICLv2",
        "image_encoder": args.image_encoder, "seed": args.seed,
        "primary_protocol_version": PRIMARY_PROTOCOL_VERSION,
        "primary_metric": primary_metric,
        "primary_value": None if args.skip_test else test_metrics[primary_metric],
        "best_epoch": best_epoch, "best_validation_value": best,
        "validation_metrics": ckpt["validation_metrics"],
        "fusion_weights": ckpt.get("fusion_weights"),
        "checkpoint": str(best_path),
        "phase1_init_checkpoint": str(phase1_init_checkpoint) if phase1_init_checkpoint is not None else None,
        "phase1_init_sha256": phase1_init_sha256,
        "projector_trainable": projector_trainable_params > 0,
        "tabicl_trainable": tabicl_trainable_params > 0,
        "projector_learning_rate": projector_lr,
        "backbone_learning_rate": args.backbone_learning_rate if args.phase == 2 else 0.0,
        "optimizer_parameter_count": optimizer_param_count,
        "duplicate_optimizer_parameters": duplicate_optimizer_params,
        "n_train": int(len(y["train"])), "n_val": int(len(y["val"])), "n_test": int(len(y["test"])),
        "test_evaluated": not args.skip_test,
        "tabular_shortcut": args.tabular_shortcut,
        "shortcut_loss_divisor": args.shortcut_loss_divisor if args.tabular_shortcut else None,
        "table_weight": ckpt.get("table_weight") if args.tabular_shortcut else None,
        "branch_test_metrics": branch_test_metrics if args.tabular_shortcut and test_metrics is not None else None,
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda_allocated_bytes = torch.cuda.max_memory_allocated(device)
        peak_cuda_reserved_bytes = torch.cuda.max_memory_reserved(device)
    else:
        peak_cuda_allocated_bytes = None
        peak_cuda_reserved_bytes = None
    result["resource_usage"] = {
        "embedding_load_seconds": embedding_load_seconds,
        "data_preprocess_seconds": data_preprocess_seconds,
        "feature_extraction_or_cache_load_seconds": feature_extraction_or_cache_load_seconds,
        "feature_peak_cuda_allocated_bytes": feature_peak_cuda_allocated_bytes,
        "feature_peak_cuda_reserved_bytes": feature_peak_cuda_reserved_bytes,
        "setup_seconds": setup_seconds,
        "train_seconds": train_seconds,
        "validation_seconds": validation_seconds,
        "total_seconds": time.monotonic() - process_started,
        "initial_cuda_allocated_bytes": initial_cuda_allocated_bytes,
        "initial_cuda_reserved_bytes": initial_cuda_reserved_bytes,
        "peak_cuda_allocated_bytes": peak_cuda_allocated_bytes,
        "peak_cuda_reserved_bytes": peak_cuda_reserved_bytes,
        "checkpoint_bytes": best_path.stat().st_size,
        "cache_bytes": {split: (Path(embedding_root) / f"{split}_{args.image_encoder}_raw.pt").stat().st_size
                        for split in ("train", "val", "test")},
        "completed_epochs": len([row for row in history if row["epoch"] >= 0]),
    }
    try:
        import resource
        result["resource_usage"]["peak_cpu_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        result["resource_usage"]["peak_cpu_rss_kib"] = None
    if test_metrics is not None:
        result.update(test_metrics)
    (run_root / "config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
    (run_root / "training_summary.json").write_text(json.dumps(history, indent=2))
    (run_root / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
