"""Dataset registry for reproducible MMPFN benchmark runs.

Adding a prepared dataset only requires one :class:`DatasetSpec` entry.  Raw
MulTaBench datasets use the official Kaggle ``data.csv + metadata.json +
images/`` contract and are converted by ``mmpfn.prepare_multabench``.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from typing import Literal


Task = Literal["classification", "regression"]
Benchmark = Literal["vtbench", "multabench", "mmpfn_paper", "custom"]
SecondaryModality = Literal["image", "text"]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    benchmark: Benchmark
    task: Task
    primary_metric: str
    higher_is_better: bool
    n_classes: int | None = None
    expected_rows: int | None = None
    expected_structured_features: int | None = None
    text_features: int = 0
    secondary_modality: SecondaryModality = "image"
    text_model_id: str | None = None
    kaggle_slug: str | None = None
    preparer_module: str | None = None
    legacy_vtbench_name: str | None = None
    # Relative to the benchmark work root (the parent of ``benchmark_data``).
    # This lets the one-command runner reuse a user-owned VT-Bench export
    # without hard-coding a machine-specific absolute path.
    legacy_data_relative_path: str | None = None
    # Portable, manifest-defined prepared-data location relative to
    # ``--data-root``.  ``{fold}`` is expanded by the suite runner.
    prepared_relative_path: str | None = None
    image_encoding: str = "image_file"
    default_max_train_context: int = 0
    target_standardization: str = "none"


def _vt(
    key: str,
    display_name: str,
    task: Task,
    *,
    legacy: str | None = None,
    image_encoding: str = "zero_one",
    n_classes: int | None = None,
    expected_rows: int | None = None,
    expected_structured_features: int | None = None,
    legacy_data_relative_path: str | None = None,
) -> DatasetSpec:
    is_classification = task == "classification"
    return DatasetSpec(
        key=key,
        display_name=display_name,
        benchmark="vtbench",
        task=task,
        primary_metric="accuracy" if is_classification else "mae",
        higher_is_better=is_classification,
        n_classes=n_classes,
        expected_rows=expected_rows,
        expected_structured_features=expected_structured_features,
        legacy_vtbench_name=legacy,
        legacy_data_relative_path=legacy_data_relative_path,
        image_encoding=image_encoding,
        default_max_train_context=0,
        target_standardization="none" if key == "vt_resp_rate" else ("train_zscore" if task == "regression" else "none"),
    )


def _mt(
    key: str,
    display_name: str,
    task: Task,
    slug: str,
    rows: int,
    features: int,
    classes: int | None = None,
) -> DatasetSpec:
    return DatasetSpec(
        key=key,
        display_name=display_name,
        benchmark="multabench",
        task=task,
        primary_metric="roc_auc" if task == "classification" else "r2",
        higher_is_better=True,
        n_classes=classes,
        expected_rows=rows,
        expected_structured_features=features,
        text_features=0,
        kaggle_slug=slug,
        image_encoding="image_file",
        # MulTaBench's official evaluation caps training at 10,000 per fold.
        default_max_train_context=10_000,
        target_standardization="train_zscore" if task == "regression" else "none",
    )


def _paper_image(
    key: str,
    display_name: str,
    *,
    n_classes: int | None = None,
) -> DatasetSpec:
    """Original MMPFN paper datasets with tabular and image inputs."""
    return DatasetSpec(
        key=key,
        display_name=display_name,
        benchmark="mmpfn_paper",
        task="classification",
        primary_metric="accuracy",
        higher_is_better=True,
        n_classes=n_classes,
        secondary_modality="image",
        preparer_module="mmpfn.prepare_mmpfn_image_benchmarks",
        default_max_train_context=10_000,
    )


_SPECS = [
    # VT-Bench discriminative datasets.  The six entries with ``legacy`` can
    # also read the exported files produced by the existing adapter scripts.
    _vt("vt_breast_cancer", "Breast Cancer", "classification", legacy="breast", image_encoding="imagenet_normalized", legacy_data_relative_path="raw/breast/features"),
    _vt(
        "vt_skin_cancer",
        "Skin Cancer",
        "classification",
        legacy="skin_cancer",
        image_encoding="uint8",
        # Produced by mmpfn.prepare_skin_cancer_vtbench.  The suite derives
        # the absolute path from --data-root, so it remains portable.
        legacy_data_relative_path="raw/skin_cancer/features",
    ),
    _vt("vt_infarction", "Infarction", "classification", legacy="infarction", legacy_data_relative_path="raw/infarction/features"),
    _vt("vt_pneumonia", "Pneumonia", "classification", legacy="pneumonia", image_encoding="uint8", legacy_data_relative_path="raw/mimic/pneumonia/features"),
    _vt("vt_los", "Length of Stay", "regression", legacy="los", image_encoding="uint8", legacy_data_relative_path="raw/mimic/los/features"),
    _vt("vt_resp_rate", "Respiratory Rate", "regression", legacy="rr", image_encoding="uint8", legacy_data_relative_path="raw/mimic/rr/features"),
    _vt("vt_adoption", "Adoption", "classification", legacy="adoption", legacy_data_relative_path="raw/adoption/features"),
    _vt(
        "vt_dvm_car",
        "DVM-Car",
        "classification",
        image_encoding="image_file",
        n_classes=286,
        expected_rows=176_414,
        expected_structured_features=17,
        legacy="dvm_car",
        legacy_data_relative_path="raw/dvm_car/features",
    ),
    _vt("vt_celeba", "CelebA", "classification", legacy="celeba", image_encoding="uint8", legacy_data_relative_path="raw/celeba/features"),
    _vt("vt_pawpularity", "Pawpularity", "regression", legacy="pawpularity", legacy_data_relative_path="raw/pawpularity/features"),
    _vt("vt_anime", "Anime", "regression", legacy="anime", image_encoding="uint8", legacy_data_relative_path="raw/anime/features"),
    # MulTaBench image-tabular datasets whose Text column is exactly zero in
    # Table 3 of the paper.  CBIS-DDSM is intentionally excluded because it
    # duplicates VT-Bench's Breast Cancer source dataset.
    _mt("mt_celeb_attractiveness", "Celeb Attractiveness", "classification", "multabench-celeb-attractiveness", 99_999, 39, 2),
    _mt("mt_chexpert", "CheXpert", "classification", "multabench-chexpert", 46_437, 17, 3),
    _mt("mt_glaucoma_smdg", "Glaucoma SMDG", "classification", "multabench-glaucoma-smdg", 12_449, 8, 3),
    _mt("mt_hateful_meme", "Hateful Meme", "classification", "multabench-hateful-meme", 10_000, 20, 2),
    _mt("mt_justin_instagram", "Justin Instagram", "classification", "multabench-justin-instagram", 10_319, 6, 5),
    _mt("mt_mammography_cmmd", "Mammography CMMD", "classification", "multabench-mammography-cmmd", 5_202, 4, 2),
    _mt("mt_zooscan_zooplankton", "Zooscan Plankton", "classification", "multabench-zooscan-zooplankton", 100_000, 28, 10),
    _mt("mt_amazon_bestseller", "Amazon Bestseller", "regression", "multabench-amazon-bestseller", 3_488, 4),
    _mt("mt_mango_mass", "Mango Mass", "regression", "multabench-mango-mass", 546, 2),
    _mt("mt_mkphoto_bots", "MkPhoto Bots", "regression", "multabench-mkphoto-bots", 13_748, 8),
    # Image-tabular datasets used in the original MMPFN paper. Their source
    # files are supplied explicitly to the preparer; no data are downloaded
    # automatically by the benchmark runner.
    _paper_image("mmpfn_cbis_ddsm_calc", "CBIS-DDSM (Calc)", n_classes=2),
    _paper_image("mmpfn_cbis_ddsm_mass", "CBIS-DDSM (Mass)", n_classes=2),
    _paper_image("mmpfn_petfinder_i", "PetFinder-I (T+I)", n_classes=5),
]


DATASET_REGISTRY = {spec.key: spec for spec in _SPECS}

_ALIASES = {
    "breast": "vt_breast_cancer",
    "pneumonia": "vt_pneumonia",
    "infarction": "vt_infarction",
    "rr": "vt_resp_rate",
    "resp_rate": "vt_resp_rate",
    "adoption": "vt_adoption",
    "dvm": "vt_dvm_car",
    "pawpularity": "vt_pawpularity",
    "cbis_calc": "mmpfn_cbis_ddsm_calc",
    "cbis_mass": "mmpfn_cbis_ddsm_mass",
    "petfinder_i": "mmpfn_petfinder_i",
    "petfinder": "mmpfn_petfinder_i",
}
for _spec in _SPECS:
    _ALIASES.setdefault(_spec.key.removeprefix("vt_").removeprefix("mt_"), _spec.key)


def _merged_specs(extra_specs: Iterable[DatasetSpec] = ()) -> list[DatasetSpec]:
    """Return built-ins plus caller-provided specs with duplicate protection."""
    merged = list(_SPECS)
    seen = set(DATASET_REGISTRY)
    for spec in extra_specs:
        if spec.key in seen:
            raise ValueError(
                f"Dataset key {spec.key!r} is already registered; choose a unique custom key."
            )
        seen.add(spec.key)
        merged.append(spec)
    return merged


def get_dataset_spec(
    name: str,
    extra_specs: Iterable[DatasetSpec] = (),
) -> DatasetSpec:
    """Resolve a built-in or manifest-provided dataset key."""
    specs = _merged_specs(extra_specs)
    registry = {spec.key: spec for spec in specs}
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    key = _ALIASES.get(normalized, normalized)
    try:
        return registry[key]
    except KeyError as error:
        choices = ", ".join(sorted(registry))
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {choices}") from error


def select_dataset_specs(
    selectors: list[str],
    extra_specs: Iterable[DatasetSpec] = (),
) -> list[DatasetSpec]:
    """Expand dataset keys and group selectors, preserving registry order."""
    extra_specs = tuple(extra_specs)
    specs = _merged_specs(extra_specs)
    registry = {spec.key: spec for spec in specs}
    requested: set[str] = set()
    for selector in selectors:
        normalized = selector.strip().lower().replace("-", "_")
        if normalized == "all":
            requested.update(registry)
        elif normalized == "vtbench":
            requested.update(spec.key for spec in specs if spec.benchmark == "vtbench")
        elif normalized in {"multabench", "multabench_text0", "multabench_image_text0"}:
            requested.update(spec.key for spec in specs if spec.benchmark == "multabench")
        elif normalized in {"mmpfn_paper", "paper_image", "mmpfn_image"}:
            requested.update(spec.key for spec in specs if spec.benchmark == "mmpfn_paper")
        elif normalized in {"custom", "manifest"}:
            requested.update(spec.key for spec in specs if spec.benchmark == "custom")
        else:
            requested.add(get_dataset_spec(selector, extra_specs).key)
    return [spec for spec in specs if spec.key in requested]


def registry_rows(extra_specs: Iterable[DatasetSpec] = ()) -> list[dict[str, object]]:
    return [
        {
            "key": spec.key,
            "benchmark": spec.benchmark,
            "task": spec.task,
            "display_name": spec.display_name,
            "primary_metric": spec.primary_metric,
            "expected_rows": spec.expected_rows,
            "structured_features": spec.expected_structured_features,
            "text_features": spec.text_features,
            "secondary_modality": spec.secondary_modality,
            "automatic_download": bool(spec.kaggle_slug),
        }
        for spec in _merged_specs(extra_specs)
    ]
