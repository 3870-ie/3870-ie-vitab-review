"""Frozen TabPFN v3.5 and TabFM replacement protocol."""

import gc
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from mmpfn.benchmarking.data import load_benchmark_splits

from .__main__ import KS, finite_metrics, read_json, select, write_json
from .backends import Backend
from .training import MultiLayerCLIPFusion, VisualPseudoColumnMLP


class Adapter(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.visual_pseudo_mlp = VisualPseudoColumnMLP(768, k)
        self.visual_multilayer_fusion = MultiLayerCLIPFusion(768)

    def forward(self, image):
        return self.visual_pseudo_mlp(self.visual_multilayer_fusion(image))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batch_indices(y, indices, cls, seed):
    if not cls:
        return train_test_split(indices, test_size=0.2, random_state=seed)
    try:
        return train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=y[indices]
        )
    except ValueError:
        rng = np.random.default_rng(seed)
        removable = []
        for label in np.unique(y[indices]):
            removable.extend(rng.permutation(indices[y[indices] == label])[1:])
        query = rng.permutation(removable)[: math.ceil(len(indices) * 0.2)]
        assert len(query) > 0
        return indices[~np.isin(indices, query)], query


def fuse(x, z, n):
    if z is None:
        return x
    ref = z[:n]
    return torch.cat([x, (z - ref.mean(0)) / (ref.std(0, correction=0) + 1e-6)], -1)


def project(adapter, images, device):
    if adapter is None:
        return None
    return torch.cat(
        [
            adapter(v.to(device, non_blocking=True).float().unsqueeze(1))
            for v in images.split(256)
        ]
    )


def evaluate(b, adapter, train, query, cls):
    if adapter is not None:
        adapter.eval()
    with torch.no_grad():
        tr_z = project(adapter, train.get("image"), b.device)
        q_z = project(adapter, query.get("image"), b.device)
        n = len(train["y"])
        values = []
        y = torch.as_tensor(
            train["y"], device=b.device, dtype=torch.long if cls else torch.float32
        )
        chunk = 64 if b.name == "tabfm" else 512
        for start in range(0, len(query["y"]), chunk):
            end = start + chunk
            x = torch.cat([train["x"], query["x"][start:end]]).to(b.device)
            z = None if tr_z is None else torch.cat([tr_z, q_z[start:end]])
            both = fuse(x, z, n)
            p = b.predict(both[:n], y, both[n:])
            assert torch.isfinite(p).all()
            values.append((p.softmax(-1) if cls else p).cpu().numpy())
        pred = np.concatenate(values)
        target = query["y"]
    if cls:
        result = dict(
            accuracy=float(accuracy_score(target, pred.argmax(1))),
            balanced_accuracy=float(balanced_accuracy_score(target, pred.argmax(1))),
            macro_f1=float(
                f1_score(target, pred.argmax(1), average="macro", zero_division=0)
            ),
        )
        try:
            result["roc_auc"] = float(
                roc_auc_score(target, pred[:, 1])
                if pred.shape[1] == 2
                else roc_auc_score(target, pred, multi_class="ovo", average="macro")
            )
        except ValueError:
            result["roc_auc"] = None
        return result
    return dict(
        rmse=float(mean_squared_error(target, pred) ** 0.5),
        mae=float(mean_absolute_error(target, pred)),
        r2=float(r2_score(target, pred)),
    )


def train_candidate(name, j, parts, k, folder, epochs, patience, weights, device):
    folder.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    block = 1250 if name == "tabfm" else 2500
    cls = j["task"] == "classification"
    key = "accuracy" if cls else "rmse"
    seed = j["seed"]
    seed_all(seed)
    b = Backend(name, j["task"], weights, estimators=2, seed=seed, device=device)
    a = Adapter(k).to(device)
    fusion = a.visual_multilayer_fusion
    opt = torch.optim.AdamW(
        [
            dict(
                params=list(a.visual_pseudo_mlp.parameters())
                + list(fusion.spatial_score.parameters()),
                lr=0.001,
            ),
            dict(params=[fusion.layer_logits, fusion.branch_logits], lr=0.01),
        ],
        weight_decay=0.0001,
    )
    tr = parts["train"]
    n = len(tr["y"])
    steps = max(1, math.ceil(n / block) * epochs)
    warm = max(1, int(0.1 * steps))

    def lr_scale(step):
        return (
            (step + 1) / warm
            if step < warm
            else 0.5
            * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, steps - warm))))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    best = None
    best_ep = None
    wait = 0
    history = []
    updates = 0
    write_json(
        folder / "config.json",
        dict(
            job=j,
            K=k,
            backend=name,
            epochs=epochs,
            patience=patience,
            fp32=True,
            tabular_shortcut=False,
            train_block=block,
            query_ratio=0.2,
            estimators_train=2,
            estimators_validation=2,
            estimators_test=8,
            lr_fusion=0.01,
            lr_adapter=0.001,
            weight_decay=0.0001,
            tfm_frozen=True,
            activation_checkpointing=name == "tabfm",
            query_chunk=64 if name == "tabfm" else 512,
        ),
    )
    for ep in range(epochs):
        a.train()
        losses = []
        order = np.random.default_rng(seed + ep).permutation(n)
        for start_i in range(0, n, block):
            idx = order[start_i : start_i + block]
            if len(idx) < 50:
                continue
            ctx, qry = batch_indices(tr["y"], idx, cls, seed + ep + start_i)
            ii = np.r_[ctx, qry]
            nctx = len(ctx)
            if cls:
                assert np.array_equal(np.unique(tr["y"][ctx]), np.unique(tr["y"])), (
                    "Context lost a class; preserve failure instead of remapping labels"
                )
            x = tr["x"][ii].to(device)
            z = project(a, tr["image"][ii], device)
            both = fuse(x, z, nctx)
            y = torch.as_tensor(
                tr["y"][ii], device=b.device, dtype=torch.long if cls else torch.float32
            )
            opt.zero_grad(set_to_none=True)
            p = b.predict(both[:nctx], y[:nctx], both[nctx:])
            loss = F.cross_entropy(p, y[nctx:]) if cls else F.mse_loss(p, y[nctx:])
            assert loss.requires_grad and torch.isfinite(loss)
            loss.backward()
            assert all(
                q.grad is None or torch.isfinite(q.grad).all() for q in a.parameters()
            )
            assert all(
                q.grad is None and not q.requires_grad
                for m in b.modules
                for q in m.parameters()
            )
            opt.step()
            scheduler.step()
            updates += 1
            losses.append(float(loss.detach()))
        assert losses
        val = evaluate(b, a, tr, parts["val"], cls)
        v = val[key]
        improve = best is None or (
            v > best[key] + 0.0001 if cls else v < best[key] - 0.0001
        )
        if improve:
            best, best_ep, wait = val, ep, 0
            torch.save(a.state_dict(), folder / "best_adapter.tmp")
            (folder / "best_adapter.tmp").replace(folder / "best_adapter.pt")
        else:
            wait += 1
        history.append(
            dict(
                epoch=ep,
                loss=float(np.mean(losses)),
                validation=val,
                weights=fusion.weight_snapshot(),
            )
        )
        write_json(folder / "training_summary.json", history)
        print(
            "EPOCH",
            name,
            j["dataset"],
            j["seed"],
            k,
            ep + 1,
            key,
            v,
            "wait",
            wait,
            flush=True,
        )
        if wait >= patience:
            break
    b.assert_frozen()
    out = dict(
        K=k,
        metric=key,
        best_validation_value=best[key],
        best_epoch=best_ep,
        epochs=len(history),
        optimizer_steps=updates,
        backbone_sha256=b.initial_hash,
        seconds=time.monotonic() - start,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device),
        test_evaluated=False,
    )
    write_json(folder / "metrics.json", out)
    return out


def run_replacement(args, spec, data, group, cache, clip):
    """Sweep the paper's K grid, then score the paired test inputs."""
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("The replacement protocol requires CUDA.")
    device = torch.device(args.device)
    weights = args.backend_weights.resolve(strict=True)
    name = args.backend
    cls = spec.task == "classification"
    metric = "accuracy" if cls else "rmse"
    parts = load_benchmark_splits(spec, data)
    label_encoder = LabelEncoder().fit(parts["train"].y) if cls else None

    def pack(split):
        ds = parts[split]
        y = label_encoder.transform(ds.y) if cls else ds.y
        result = {"x": torch.from_numpy(np.ascontiguousarray(ds.x)).float(), "y": y}
        result["image"] = ds.get_embeddings(
            None,
            cache / f"{split}_clip_vitl14_multilayer_raw.pt",
            16,
            str(device),
            image_encoder="clip_vitl14",
            image_model_id=str(clip),
            token_type="patch",
            clip_spatial_patch_layers=(6, 12, 18, 24),
        )
        ds.embeddings = None
        return result

    job = {"dataset": spec.key, "seed": args.seed, "task": spec.task}
    selected_file = group / "selected.json"
    if selected_file.is_file():
        return
    training = {split: pack(split) for split in ("train", "val")}
    candidates = {}
    for k in KS:
        folder = group / f"k{k}"
        result = folder / "metrics.json"
        checkpoint = folder / "best_adapter.pt"
        if not result.is_file():
            train_candidate(name, job, training, k, folder, 50, 15, weights, device)
            gc.collect()
            torch.cuda.empty_cache()
        record = read_json(result)
        if (
            record["K"] != k
            or record["metric"] != metric
            or record["test_evaluated"]
            or not checkpoint.is_file()
        ):
            raise ValueError(f"Invalid validation candidate: {folder}")
        candidates[k] = record["best_validation_value"]

    chosen = select(candidates, metric)
    adapter = Adapter(chosen).to(device)
    adapter.load_state_dict(
        torch.load(
            group / f"k{chosen}" / "best_adapter.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    test = pack("test")
    backend = Backend(
        name, spec.task, weights, estimators=8, seed=args.seed, device=device
    )
    multi = evaluate(backend, adapter, training["train"], test, cls)
    backend.assert_frozen()
    del backend
    gc.collect()
    torch.cuda.empty_cache()
    backend = Backend(
        name, spec.task, weights, estimators=8, seed=args.seed, device=device
    )
    table = evaluate(backend, None, training["train"], test, cls)
    backend.assert_frozen()
    write_json(
        selected_file,
        {
            "dataset": spec.key,
            "seed": args.seed,
            "selected_k": chosen,
            "validation_metric": metric,
            "validation": candidates,
            "test": finite_metrics(multi),
            "table_only": finite_metrics(table),
            "gain": (1 if cls else -1) * (multi[metric] - table[metric]),
        },
    )
