# ViTab-Adapter

Anonymous review repository for **ViTab-Adapter: Lightweight Visual Adaptation
for Tabular Foundation Models**. ViTab-Adapter maps frozen visual features to
a small set of pseudo-columns and appends them to the original table for
prediction with a frozen tabular foundation model.

## What is included

- Dataset preparation for the VT-Bench and MulTaBench image–table experiments.
- Training with frozen CLIP ViT-L/14 and TabICLv2, validation-based selection
  of the pseudo-column count, and test evaluation of the selected checkpoint.
- Patch-count sensitivity, test-time modality ablation, and visual encoder
  and tabular backend comparisons.
- Per-seed metrics and CSV summaries across three seeds.

## Repository layout

| Path | Purpose |
| --- | --- |
| `vitab_adapter/__main__.py` | Experiment and summary commands |
| `vitab_adapter/prepare.py` | Benchmark split preparation |
| `vitab_adapter/training.py` | Visual adapter and TabICLv2 training and evaluation |
| `vitab_adapter/vision.py` | Alternative vision encoders |
| `vitab_adapter/backends.py`, `vitab_adapter/replacement.py` | Frozen tabular backend comparisons |
| `mmpfn/` | Reused benchmark data and preprocessing helpers |

## Installation

Run the commands from the repository root with Python 3.11 and a CUDA-enabled
PyTorch environment.

Install the CUDA 12.4 builds of PyTorch 2.5.1 and torchvision 0.20.1, then run
`python -m pip install -r requirements.txt`. Provide a local Hugging Face
snapshot of `openai/clip-vit-large-patch14` and the task-appropriate checkpoint
from `jingang/TabICL`: `tabicl-classifier-v2-20260212.ckpt` or
`tabicl-regressor-v2-20260212.ckpt`. Keep the model snapshots unchanged across runs.

## Prepare data

Obtain the datasets from their official sources and follow the applicable
access terms. For VT-Bench, supply the benchmark's preprocessed feature
exports at `DATA_BASE/vt/<legacy_name>/features`. For MulTaBench, supply the
downloaded dataset exports at `DATA_BASE/multabench/_downloads/<slug>`, with
`data.csv`, `metadata.json` and the referenced images. The metadata identifies
the image and target columns via `image_col` and `target`. Supported dataset
IDs and slugs are listed in the preparation script and registry.

```sh
python -m vitab_adapter.prepare --data-base DATA_BASE --output prepared
```

The preparation command also accepts repeated `--dataset` flags. It copies
the existing VT-Bench split and uses the TabSTAR 1.1.15 split implementation
for the aligned MulTaBench folds. All three seeds use VT-Bench fold 0.
For MulTaBench, seeds 42, 43 and 44 use folds 0, 1 and 2, respectively. For exact reproduction, supply
the same source dataset snapshots and filtering used by the experiments.

The runner consumes `prepared/<dataset>/fold_<n>/metadata.json` and
`train.npz`, `val.npz`, `test.npz`. Each NPZ contains `x`, `y` and
`image_paths` in matching row order. Relative paths are resolved against
`image_base_dir` in metadata. Prepared files may reference local licensed
images and should stay outside the review repository.

## Quick start

```sh
python -m vitab_adapter run --dataset vt_skin_cancer --seed 42 --prepared prepared --output runs --clip-model weights/clip --tabicl-checkpoint weights/tabicl-classifier-v2-20260212.ckpt
python -m vitab_adapter summarize --output runs --csv summary.csv
```

Repeat the training command for seeds 43 and 44. The runner trains six
candidates (K = 4, 8, 16, 32, 64, 128) and uses validation accuracy
(classification) or RMSE (regression) to select K. Ties select the smaller K.
Only the selected checkpoint is evaluated on the test split. The CSV reports
means and sample standard deviations for metrics with all three seeds
available. Accuracy is expressed as a percentage and AUROC and macro-F1
are expressed as fractions.

Patch features come from CLIP blocks 6, 12, 18 and 24. A 2×2 average pool
retains 64 spatial positions per layer, and the global CLS feature comes from
block 24. Spatial attention uses hidden width 128. The pseudo-column MLP
uses hidden width 512 and dropout 0.1. AdamW uses learning rates 0.01 for
the fusion logits and 0.001 for the rest of the adapter, weight decay
0.0001, cosine decay with 10% warmup and a gradient norm cap of 1.
Training uses up to 50 epochs, patience 15, at most 2,500 rows per
training block and query fraction 0.2. The train/validation/test ensemble
sizes are 2/2/8.

Run only one process per dataset/seed/output combination. Completed
candidates are reused on rerun. CLIP feature caches are shared across
seeds that use the same prepared split, and are keyed by the prepared
split files and CLIP snapshot path. Use a new output root if source
images or weights change in place.

## Ablations and encoder replacements

Add the following options to the training command above. Use the same
prepared split and TabICLv2 checkpoint for the comparisons.

| Experiment | Options | Protocol |
| --- | --- | --- |
| E3 | `--variant E3` (default) | Standard configuration with 64 patches per layer. |
| E1 / E2 | `--variant E1` or `--variant E2` | Pool the projected 8×8 patch grid to 1 or 16 tokens per layer. |
| E4 | `--variant E4` | Use the original 256 patches per layer. Add `--train-block 1250` for the smaller-block runs. |
| DINOv2 | `--vision dinov2 --clip-model weights/dinov2` | Use the local snapshot of `facebook/dinov2-large`. |
| OpenCLIP | `--vision openclip --clip-model weights/openclip` | Use the local Transformers snapshot of `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`. |
| Test-time modality ablation | `--modality-ablation` | Reuse the selected standard CLIP + TabICLv2 checkpoint. Replace the corresponding input columns with per-estimator training-context means before TabICLv2, preserving feature order and width. |

The standard run writes into `runs`. Other configurations write into
`runs/<variant-or-encoder>`. Summarize each configuration using its own
output root. For modality ablation, run
`python -m vitab_adapter summarize --output runs --kind modality --csv modality.csv`.

## Tabular backend replacements

Install the compatible optional backend in its own environment with the
base requirements. For TabPFN use the v3.5-capable TabPFN 8.2.0 runtime and
the v3.5 checkpoint. For TabFM install the official Google Research TabFM
PyTorch source exposing `tabfm.src.pytorch.tabfm_v1_0_0.load` and use the
local TabFM v1.0.0 checkpoint directory containing its config and weights.

```sh
python -m vitab_adapter run --dataset vt_skin_cancer --seed 42 --prepared prepared --output runs --clip-model weights/clip --backend tabpfn35 --backend-weights weights/tabpfn-v3.5-20260909.safetensors
python -m vitab_adapter run --dataset vt_skin_cancer --seed 42 --prepared prepared --output runs --clip-model weights/clip --backend tabfm --backend-weights weights/tabfm
python -m vitab_adapter summarize --output runs/tabfm --csv tabfm.csv
```

The replacement workflow preserves the backend-specific experimental
settings. TabPFN v3.5 uses 2,500 rows per training block and 512 query rows
per evaluation chunk. TabFM uses 1,250 rows per training block, 64 query
rows per evaluation chunk, and activation checkpointing for its attention
blocks. Both freeze the predictor, select K on validation, and evaluate
the selected multimodal adapter alongside a paired table-only prediction
on the same test split. The summary includes the paired gain, defined as
multimodal minus table-only ACC or table-only minus multimodal RMSE.

## Outputs and data access

The run directory contains per-candidate validation metrics, selected test
metrics, and local feature caches and checkpoints. Download datasets and model
weights from their upstream providers under their access terms. The review
repository contains source code and documentation. Keep dataset files,
checkpoints, caches, logs, and credentials outside Git.

## Third-party code and attribution

The benchmark loading and preprocessing helpers under `mmpfn` are
adapted from MultiModalPFN. The accompanying third-party license is
included in [LICENSE-MultiModalPFN](LICENSE-MultiModalPFN).

Built with PriorLabs-TabPFN (attribution for the reused benchmark helpers).
