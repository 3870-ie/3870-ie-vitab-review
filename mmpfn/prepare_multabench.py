"""Download and prepare official MulTaBench image-tabular datasets for MMPFN.

The official Kaggle release stores every dataset as ``data.csv``,
``metadata.json`` and ``images/``.  This adapter creates deterministic outer
train/test folds and an inner validation split without modifying the source.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from mmpfn.benchmarking.registry import DatasetSpec, select_dataset_specs


KAGGLE_OWNER = "chico89"


def _parse_mapping(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected DATASET=PATH, received: {value}")
        key, path = value.split("=", 1)
        result[key.strip()] = Path(path).expanduser().resolve()
    return result


def _download(spec: DatasetSpec, data_root: Path) -> Path:
    if not spec.kaggle_slug:
        raise ValueError(f"{spec.key} has no Kaggle source configured.")
    cache_root = data_root / "_kagglehub"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(cache_root)
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError("Install kagglehub first: python -m pip install kagglehub") from error

    handle = f"{KAGGLE_OWNER}/{spec.kaggle_slug}"
    kwargs: dict[str, str] = {}
    if "output_dir" in inspect.signature(kagglehub.dataset_download).parameters:
        target = data_root / "_downloads" / spec.kaggle_slug
        target.mkdir(parents=True, exist_ok=True)
        kwargs["output_dir"] = str(target)
    print(f"Downloading official MulTaBench dataset {handle} ...", flush=True)
    return Path(kagglehub.dataset_download(handle, **kwargs)).expanduser().resolve()


def _find_release_root(downloaded: Path) -> Path:
    candidates = [downloaded, *[path.parent for path in downloaded.rglob("metadata.json")]]
    for candidate in candidates:
        if (candidate / "metadata.json").is_file() and (candidate / "data.csv").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not locate MulTaBench metadata.json and data.csv below {downloaded}"
    )


def _resolve_image(source_root: Path, value: object) -> Path | None:
    if pd.isna(value):
        return None
    raw = Path(str(value))
    candidates = [raw] if raw.is_absolute() else [source_root / raw, source_root / "images" / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _safe_split(
    indices: np.ndarray,
    y: np.ndarray,
    *,
    test_size: int,
    seed: int,
    classification: bool,
) -> tuple[np.ndarray, np.ndarray]:
    stratify = y if classification else None
    try:
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError as error:
        print(f"WARNING: stratified split failed ({error}); using deterministic unstratified split.")
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )


def _metadata_categorical_columns(meta: dict, feature_columns: list[str]) -> set[str]:
    for key in ("categorical_columns", "categorical_cols", "cat_cols"):
        values = meta.get(key)
        if values:
            return {str(value) for value in values if str(value) in feature_columns}
    return set()


def _fit_transform_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    train_indices: np.ndarray,
    metadata: dict,
) -> tuple[np.ndarray, list[int], list[int], dict[str, object]]:
    explicit_categorical = _metadata_categorical_columns(metadata, feature_columns)
    output = np.zeros((len(frame), len(feature_columns)), dtype=np.float32)
    categorical_indices: list[int] = []
    field_lengths: list[int] = []
    preprocessing: dict[str, object] = {}
    for column_index, column in enumerate(feature_columns):
        series = frame[column]
        categorical = (
            column in explicit_categorical
            or is_bool_dtype(series.dtype)
            or not is_numeric_dtype(series.dtype)
        )
        if categorical:
            train_values = series.iloc[train_indices].fillna("<MISSING>").astype(str)
            categories = sorted(train_values.unique().tolist())
            mapping = {value: index + 1 for index, value in enumerate(categories)}
            output[:, column_index] = (
                series.fillna("<MISSING>").astype(str).map(mapping).fillna(0).to_numpy(dtype=np.float32)
            )
            categorical_indices.append(column_index)
            field_lengths.append(len(mapping) + 1)
            preprocessing[column] = {"kind": "categorical", "unknown_code": 0, "mapping": mapping}
        else:
            numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
            median = float(numeric.iloc[train_indices].median())
            if not np.isfinite(median):
                median = 0.0
            output[:, column_index] = numeric.fillna(median).to_numpy(dtype=np.float32)
            field_lengths.append(1)
            preprocessing[column] = {"kind": "numeric", "train_median": median}
    return output, categorical_indices, field_lengths, preprocessing


def prepare_dataset_fold(
    spec: DatasetSpec,
    *,
    data_root: Path,
    fold: int,
    source_root: Path | None = None,
    force: bool = False,
) -> Path:
    output = data_root / spec.key / f"fold_{fold}"
    if (output / "metadata.json").is_file() and all((output / f"{split}.npz").is_file() for split in ("train", "val", "test")):
        if not force:
            print(f"Prepared data already exists, skipping: {output}")
            return output

    source_root = _find_release_root(source_root or _download(spec, data_root))
    source_meta = json.loads((source_root / "metadata.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(source_root / "data.csv")
    target_column = source_meta["target"]
    image_column = source_meta["image_col"]
    if target_column not in frame or image_column not in frame:
        raise ValueError(
            f"{spec.key}: metadata columns target={target_column!r}, image_col={image_column!r} "
            "are not both present in data.csv"
        )

    feature_columns = [column for column in frame.columns if column not in {target_column, image_column}]
    if spec.expected_structured_features is not None and len(feature_columns) != spec.expected_structured_features:
        print(
            f"WARNING: {spec.key} contains {len(feature_columns)} structured columns; "
            f"the paper reports {spec.expected_structured_features}."
        )

    image_paths = [_resolve_image(source_root, value) for value in frame[image_column]]
    target_valid = frame[target_column].notna().to_numpy()
    image_valid = np.asarray([path is not None and path.is_file() for path in image_paths])
    keep = target_valid & image_valid
    if not np.all(keep):
        print(f"WARNING: dropping {int((~keep).sum())} rows with missing target/image in {spec.key}.")
        frame = frame.loc[keep].reset_index(drop=True)
        image_paths = [path for path, valid in zip(image_paths, keep) if valid]

    if spec.task == "classification":
        encoder = LabelEncoder()
        y = encoder.fit_transform(frame[target_column].astype(str)).astype(np.int64)
        label_mapping: dict[str, object] = {
            str(index): str(label) for index, label in enumerate(encoder.classes_.tolist())
        }
    else:
        y = pd.to_numeric(frame[target_column], errors="raise").to_numpy(dtype=np.float32)
        label_mapping = {}

    indices = np.arange(len(frame))
    test_count = min(2_000, max(1, math.ceil(0.10 * len(indices))))
    train_val, test = _safe_split(
        indices,
        y,
        test_size=test_count,
        seed=fold,
        classification=spec.task == "classification",
    )
    val_count = max(1, math.ceil(0.10 * len(train_val)))
    train_positions, val_positions = _safe_split(
        np.arange(len(train_val)),
        y[train_val],
        test_size=val_count,
        seed=10_000 + fold,
        classification=spec.task == "classification",
    )
    train = train_val[train_positions]
    val = train_val[val_positions]

    x, categorical_indices, field_lengths, preprocessing = _fit_transform_features(
        frame,
        feature_columns,
        train,
        source_meta,
    )
    output.mkdir(parents=True, exist_ok=True)
    for split_name, split_indices in (("train", train), ("val", val), ("test", test)):
        np.savez(
            output / f"{split_name}.npz",
            x=x[split_indices],
            y=y[split_indices],
            image_paths=np.asarray([str(image_paths[index]) for index in split_indices], dtype=str),
        )

    prepared_meta = {
        "format_version": 1,
        "dataset": spec.key,
        "display_name": spec.display_name,
        "benchmark": spec.benchmark,
        "task": spec.task,
        "fold": fold,
        "split_protocol": {
            "outer_test": "10%, capped at 2000; stratified for classification",
            "inner_validation": "10% of outer training split; stratified for classification",
            "outer_seed": fold,
            "inner_seed": 10_000 + fold,
        },
        "source_root": str(source_root),
        "image_base_dir": str(source_root),
        "image_encoding": "image_file",
        "target_column": target_column,
        "image_column": image_column,
        "feature_columns": feature_columns,
        "categorical_indices": categorical_indices,
        "field_lengths": field_lengths,
        "label_mapping": label_mapping,
        "preprocessing": preprocessing,
        "counts": {"all": len(frame), "train": len(train), "val": len(val), "test": len(test)},
    }
    (output / "metadata.json").write_text(json.dumps(prepared_meta, indent=2), encoding="utf-8")
    print(
        f"Prepared {spec.key} fold {fold}: train={len(train)}, val={len(val)}, "
        f"test={len(test)}, features={x.shape[1]}, categorical={len(categorical_indices)}"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["multabench_text0"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Use an already-downloaded official release instead of Kaggle.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    specs = select_dataset_specs(args.datasets)
    invalid = [spec.key for spec in specs if spec.benchmark != "multabench"]
    if invalid:
        raise ValueError(f"prepare_multabench only accepts MulTaBench datasets: {invalid}")
    source_roots = _parse_mapping(args.source_root)
    data_root = args.data_root.expanduser().resolve()
    for spec in specs:
        for fold in args.folds:
            prepare_dataset_fold(
                spec,
                data_root=data_root,
                fold=fold,
                source_root=source_roots.get(spec.key),
                force=args.force,
            )


if __name__ == "__main__":
    main()
