"""Run the paper protocol and aggregate completed, validation-selected tests."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

KS = (4, 8, 16, 32, 64, 128)
SEEDS = (42, 43, 44)
METRICS = ("accuracy", "roc_auc", "macro_f1", "rmse", "mae", "r2")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def select(candidates, metric):
    if set(candidates) != set(KS) or not all(
        math.isfinite(v) for v in candidates.values()
    ):
        raise ValueError("Selection requires finite scores for all candidates.")
    if metric not in ("accuracy", "rmse"):
        raise ValueError(f"Unexpected selection metric {metric}")
    sign = -1 if metric == "accuracy" else 1
    return min(KS, key=lambda k: (sign * candidates[k], k))


def run(args):
    from mmpfn.benchmarking.registry import get_dataset_spec

    spec = get_dataset_spec(args.dataset)
    fold = args.seed - 42 if spec.benchmark == "multabench" else 0
    data = (args.prepared / spec.key / f"fold_{fold}").resolve()
    variant = args.variant
    if args.vision != "clip" and variant != "E3":
        raise ValueError(
            "Run alternative vision encoders with the standard patch count."
        )
    if args.modality_ablation and (variant != "E3" or args.vision != "clip"):
        raise ValueError("Modality ablation uses the selected CLIP checkpoint.")
    if args.train_block is not None and variant != "E4":
        raise ValueError("--train-block is reserved for E4.")
    encoder = "clip_vitl14_multilayer"
    replacement = args.backend != "tabicl"
    if replacement and (
        variant != "E3" or args.vision != "clip" or args.modality_ablation
    ):
        raise ValueError("Backend replacement uses the standard CLIP adapter.")
    weights = args.backend_weights if replacement else args.tabicl_checkpoint
    if weights is None:
        raise ValueError(
            "Supply --backend-weights for replacement or --tabicl-checkpoint for TabICLv2."
        )
    model = weights.resolve(strict=True)
    clip = Path(args.clip_model).resolve(strict=True)
    if not (clip / "config.json").is_file():
        raise ValueError(f"Expected a local Hugging Face vision model snapshot: {clip}")
    metadata_path = data / "metadata.json"
    metadata = read_json(metadata_path)
    if metadata["dataset"] != spec.key or metadata["fold"] != fold:
        raise ValueError(f"Prepared split identity does not match {data}")
    split_files = [
        metadata_path,
        *(data / f"{name}.npz" for name in ("train", "val", "test")),
    ]
    split_hashes = {path.name: digest(path) for path in split_files}
    cache_root = args.output.resolve() / "clip_cache"
    output = args.output.resolve()
    standard = not replacement and variant == "E3" and args.vision == "clip"
    if not standard:
        output = output / (
            args.backend
            if replacement
            else variant
            if args.vision == "clip"
            else args.vision
        )
    group = output / spec.key / f"seed_{args.seed}"
    group.mkdir(parents=True, exist_ok=True)
    identity = {
        "dataset": spec.key,
        "seed": args.seed,
        "fold": fold,
        "split_files": split_hashes,
        "clip_model": str(clip),
        "data_root": str(data),
    }
    if replacement:
        identity["backend"] = args.backend
        identity["backend_weights"] = (
            {
                path.relative_to(model).as_posix(): digest(path)
                for path in sorted(model.rglob("*"))
                if path.is_file()
            }
            if model.is_dir()
            else {model.name: digest(model)}
        )
    else:
        identity["tabicl_checkpoint_sha256"] = digest(model)
    if not standard and not replacement:
        identity.update(
            variant=variant, vision=args.vision, train_block=args.train_block or 2500
        )
    manifest = group / "run.json"
    if manifest.exists() and read_json(manifest) != identity:
        raise ValueError(f"Run inputs changed. Use a new output directory: {group}")
    write_json(manifest, identity)
    cache_key = hashlib.sha256(
        json.dumps([str(data), split_hashes, str(clip)], sort_keys=True).encode()
    ).hexdigest()
    cache = cache_root / spec.key / cache_key
    if variant == "E4" or args.vision != "clip":
        cache /= variant if args.vision == "clip" else args.vision
    if replacement:
        from .replacement import run_replacement

        run_replacement(args, spec, data, group, cache, clip)
        return
    metric = "accuracy" if spec.task == "classification" else "rmse"
    common = [
        sys.executable,
        "-u",
        "-m",
        "vitab_adapter.training",
        "--dataset",
        spec.key,
        "--data-root",
        str(data),
        "--output-dir",
        str(group),
        "--embedding-root",
        str(cache),
        "--device",
        args.device,
        "--image-encoder",
        encoder,
        "--variant",
        variant,
        "--vision",
        args.vision,
        "--image-model-id",
        str(clip),
        "--model-path",
        str(model),
        "--seed",
        str(args.seed),
        "--fusion-mode",
        "pseudo_mlp",
        "--phase",
        "1",
        "--epochs",
        "50",
        "--patience",
        "15",
        "--max-data-size",
        str(args.train_block or 2500),
        "--query-ratio",
        "0.2",
        "--warmup-proportion",
        "0.1",
        "--grad-clip",
        "1.0",
        "--projector-learning-rate",
        "0.001",
        "--projector-weight-decay",
        "0.0001",
        "--dropout",
        "0.1",
        "--n-estimators-finetune",
        "2",
        "--n-estimators-validation",
        "2",
        "--n-estimators-inference",
        "8",
    ]

    def candidate_path(k):
        prefix = group / spec.key / f"fold_{fold}" / "pseudo_mlp"
        prefix /= encoder
        return prefix / f"k{k}" / "phase1" / f"seed_{args.seed}"

    candidates = {}
    for k in KS:
        directory = candidate_path(k)
        result = directory / "metrics.json"
        checkpoint = directory / "best_pseudo_columns.pt"
        if not result.is_file():
            subprocess.run(
                [
                    *common,
                    "--pseudo-dim",
                    str(k),
                    "--skip-test",
                ],
                check=True,
            )
        metrics = read_json(result)
        if (
            metrics["test_evaluated"]
            or metrics["tabicl_trainable"]
            or metrics["dataset"] != spec.key
            or metrics["primary_metric"] != metric
            or metrics["seed"] != args.seed
            or metrics["pseudo_dim"] != k
            or not checkpoint.is_file()
        ):
            raise ValueError(f"Invalid validation candidate: {directory}")
        candidates[k] = metrics["best_validation_value"]

    k = select(candidates, metric)
    directory = candidate_path(k)
    test_path = directory / "test_metrics.json"
    if not test_path.is_file():
        subprocess.run(
            [
                *common,
                "--pseudo-dim",
                str(k),
                "--test-only",
                "--checkpoint",
                str(directory / "best_pseudo_columns.pt"),
            ],
            check=True,
        )
    test = read_json(test_path)
    if (
        not test["test_evaluated"]
        or test["dataset"] != spec.key
        or test["primary_metric"] != metric
        or test["seed"] != args.seed
        or test["pseudo_dim"] != k
        or not math.isfinite(test["primary_value"])
    ):
        raise ValueError(f"Invalid selected test result: {test_path}")
    if args.modality_ablation:
        masked = {}
        for mode in ("without-image", "without-table"):
            target = directory / f"test_{mode}.json"
            if not target.is_file():
                subprocess.run(
                    [
                        *common,
                        "--pseudo-dim",
                        str(k),
                        "--test-only",
                        "--modality",
                        mode,
                        "--checkpoint",
                        str(directory / "best_pseudo_columns.pt"),
                    ],
                    check=True,
                )
            result = read_json(target)
            if (
                not result["test_evaluated"]
                or result["dataset"] != spec.key
                or result["seed"] != args.seed
                or result["pseudo_dim"] != k
                or result["primary_metric"] != metric
                or not math.isfinite(result["primary_value"])
            ):
                raise ValueError(f"Invalid modality test result: {target}")
            masked[mode] = finite_metrics(result)
        write_json(
            group / "modality.json",
            {
                "dataset": spec.key,
                "seed": args.seed,
                "selected_k": k,
                "full": finite_metrics(test),
                **masked,
            },
        )
    write_json(
        group / "selected.json",
        {
            "dataset": spec.key,
            "seed": args.seed,
            "fold": fold,
            "selected_k": k,
            "validation_metric": metric,
            "validation": candidates,
            "test": finite_metrics(test),
        },
    )


def finite_metrics(result):
    return {
        name: result[name]
        if result[name] is not None and math.isfinite(result[name])
        else None
        for name in METRICS
        if name in result
    }


def summarize(args):
    datasets = {}
    for path in sorted(args.output.glob(f"*/seed_*/{args.kind}.json")):
        record = read_json(path)
        group = datasets.setdefault(record["dataset"], {})
        seed = record["seed"]
        if seed in group:
            raise ValueError(f"Duplicate dataset and seed: {path}")
        group[seed] = record
    if not datasets:
        raise SystemExit("No validation-selected tests found.")
    with args.csv.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(
            [
                "dataset",
                "condition",
                "metric",
                "n_seeds",
                "mean",
                "sample_sd",
                "K_seed42",
                "K_seed43",
                "K_seed44",
            ]
        )
        for dataset, records in sorted(datasets.items()):
            keys = [records[s]["selected_k"] if s in records else "" for s in SEEDS]
            conditions = (
                ("full", "without-image", "without-table")
                if args.kind == "modality"
                else ("test", "table_only")
            )
            for condition in conditions:
                for metric in METRICS:
                    values = [
                        records[s].get(condition, {}).get(metric)
                        for s in SEEDS
                        if s in records
                    ]
                    values = [v for v in values if v is not None]
                    if not any(
                        metric in r.get(condition, {}) for r in records.values()
                    ):
                        continue
                    write_summary_row(writer, dataset, condition, metric, values, keys)
            gains = [
                records[s]["gain"]
                for s in SEEDS
                if s in records and "gain" in records[s]
            ]
            if gains:
                metric = next(iter(records.values()))["validation_metric"]
                write_summary_row(writer, dataset, "paired_gain", metric, gains, keys)


def write_summary_row(writer, dataset, condition, metric, values, keys):
    mean = sd = ""
    if len(values) == len(SEEDS):
        scale = 100 if metric == "accuracy" else 1
        mean = statistics.mean(values) * scale
        sd = statistics.stdev(values) * scale
    writer.writerow([dataset, condition, metric, len(values), mean, sd, *keys])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    experiment = commands.add_parser("run", help="Run one dataset and seed")
    experiment.add_argument("--dataset", required=True)
    experiment.add_argument("--seed", type=int, choices=SEEDS, required=True)
    experiment.add_argument("--prepared", type=Path, required=True)
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument(
        "--clip-model",
        required=True,
        help="Local snapshot of the selected vision encoder",
    )
    experiment.add_argument("--tabicl-checkpoint", type=Path)
    experiment.add_argument(
        "--backend", choices=("tabicl", "tabpfn35", "tabfm"), default="tabicl"
    )
    experiment.add_argument("--backend-weights", type=Path)
    experiment.add_argument("--device", default="cuda:0")
    experiment.add_argument(
        "--variant",
        choices=("E1", "E2", "E3", "E4"),
        default="E3",
    )
    experiment.add_argument(
        "--vision", choices=("clip", "dinov2", "openclip"), default="clip"
    )
    experiment.add_argument("--train-block", type=int, choices=(1250, 2500))
    experiment.add_argument("--modality-ablation", action="store_true")
    experiment.set_defaults(handler=run)
    aggregate = commands.add_parser("summarize")
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--csv", type=Path, required=True)
    aggregate.add_argument(
        "--kind", choices=("selected", "modality"), default="selected"
    )
    aggregate.set_defaults(handler=summarize)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
