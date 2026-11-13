# ViTab-Adapter

## 1. Introduction

ViTab-Adapter is a lightweight method for prediction tasks that combine images and structured data. It converts features from a frozen vision encoder into a small set of **visual pseudo-columns** and appends them to the original table. A frozen tabular foundation model makes predictions from the augmented table, while training updates only the visual adapter.

This repository provides data preparation, training, and evaluation commands for 14 classification and regression datasets from VT-Bench and MulTaBench. The default configuration uses CLIP ViT-L/14 and TabICLv2.

## 2. Overview

A single global image vector can miss localized details, while passing all patch tokens into the predictor makes the input much larger. ViTab-Adapter combines global and local information before converting it to a compact tabular representation.

1. **Multi-layer fusion.** The adapter pools patch grids from CLIP layers 6, 12, 18, and 24, then combines corresponding positions using learned layer weights.
2. **Spatial aggregation and CLS fusion.** A scoring network weights spatial positions to form a local representation. The adapter combines it with the final-layer CLS representation using learned weights.
3. **Pseudo-column generation.** An MLP maps the fused vector to K scalar features. The adapter standardizes them using training-context statistics and appends them to the original columns for prediction.

![ViTab-Adapter architecture, multi-layer fusion, and spatial aggregation](assets/overview.png)

## 3. Quickstart

### 3.1 Download

On GitHub, select **Code → Download ZIP** and extract the archive. Run the following commands from the extracted repository root.

### 3.2 Set up the environment

Use Python 3.11 with CUDA-enabled PyTorch. Install the CUDA 12.4 builds of PyTorch 2.5.1 and torchvision 0.20.1, then run:

```bash
python -m pip install -r requirements.txt
```

Prepare a local snapshot of `openai/clip-vit-large-patch14` and the TabICLv2 checkpoint for your task:

- **Classification** — `tabicl-classifier-v2-20260212.ckpt`
- **Regression** — `tabicl-regressor-v2-20260212.ckpt`

Use the same model snapshots across runs.

### 3.3 Prepare the data

Obtain the datasets from their official sources and follow their access and usage terms.

| Benchmark | Expected location | Required files |
| --- | --- | --- |
| VT-Bench | `DATA_BASE/vt/<legacy_name>/features` | Preprocessed benchmark features |
| MulTaBench | `DATA_BASE/multabench/_downloads/<slug>` | `data.csv`, `metadata.json`, and the referenced images |

In the MulTaBench metadata, `image_col` identifies the image column and `target` identifies the prediction target. See the [dataset registry](mmpfn/benchmarking/registry.py) and [preparation script](vitab_adapter/prepare.py) for supported dataset IDs and slugs.

```bash
python -m vitab_adapter.prepare --data-base DATA_BASE --output prepared
```

Pass `--dataset` multiple times to prepare a subset of datasets.

### 3.4 Train and evaluate

```bash
python -m vitab_adapter run --dataset vt_skin_cancer --seed 42 --prepared prepared --output runs --clip-model weights/clip --tabicl-checkpoint weights/tabicl-classifier-v2-20260212.ckpt
python -m vitab_adapter summarize --output runs --csv summary.csv
```

- **Seeds** — Run 42, 43, and 44.
- **Model selection** — Sweep K = 4, 8, 16, 32, 64, and 128. Select by validation accuracy for classification or validation RMSE for regression. Break ties in favor of the smaller K and evaluate the selected checkpoint on the test split.
- **Reporting** — Report test mean ± sample standard deviation across the three seeds.
- **Training** — Use up to 50 epochs, early stopping patience of 15, and up to 2,500 rows per training block. Ensemble sizes are 2 for training, 2 for validation, and 8 for testing.
- **Reruns** — Completed candidates and frozen features are reused. Use a new output directory if images or model weights change in place.

## 4. Ablations and vision encoder replacements

Add the relevant options to the training command above. Use the same data split and TabICLv2 checkpoint for each comparison.

| Setting | Options | Protocol |
| --- | --- | --- |
| 64 patches per layer (default) | `--variant E3` | Use the standard patch count. |
| 1 or 16 patches per layer | `--variant E1` or `--variant E2` | Pool the projected 8×8 patch grid to 1 or 16 tokens per layer. |
| 256 patches per layer | `--variant E4` | Use all 256 patches. Add `--train-block 1250` for the smaller-block runs. |
| DINOv2 | `--vision dinov2 --clip-model weights/dinov2` | Use a local snapshot of `facebook/dinov2-large`. |
| OpenCLIP | `--vision openclip --clip-model weights/openclip` | Use a local Transformers snapshot of `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`. |
| Test-time modality ablation | `--modality-ablation` | Reuse the selected CLIP + TabICLv2 checkpoint. Replace the corresponding input columns with each estimator's training-context means before TabICLv2, preserving column order and width. |

The standard configuration writes to `runs`. Other patch counts and vision encoders write to `runs/<variant-or-encoder>`.

## 5. Tabular backend replacements

Install each optional backend in a separate Python environment with the base dependencies.

- **TabPFN v3.5** — Use TabPFN 8.2.0 and the v3.5 checkpoint.
- **TabFM** — Install the Google Research PyTorch source exposing `tabfm.src.pytorch.tabfm_v1_0_0.load`. Use a local TabFM v1.0.0 checkpoint directory containing the model configuration and weights.

```bash
python -m vitab_adapter run --dataset vt_skin_cancer --seed 42 --prepared prepared --output runs --clip-model weights/clip --backend tabpfn35 --backend-weights weights/tabpfn-v3.5-20260909.safetensors
python -m vitab_adapter run --dataset vt_skin_cancer --seed 42 --prepared prepared --output runs --clip-model weights/clip --backend tabfm --backend-weights weights/tabfm
```

| Backend | Rows per training block | Query rows per evaluation chunk |
| --- | ---: | ---: |
| TabPFN v3.5 | 2,500 | 512 |
| TabFM | 1,250 | 64 |

TabFM uses activation checkpointing for attention blocks.

## 6. Results

ViTab-Adapter with CLIP and TabICLv2 was evaluated on 14 datasets using aligned splits. Scores are test mean ± sample standard deviation over seeds 42, 43, and 44.

### Classification · Accuracy (%) ↑

| Dataset | ViTab-Adapter |
| --- | ---: |
| Skin Cancer | 86.96 ± 0.00 |
| Breast Cancer | 53.95 ± 1.79 |
| Infarction | 71.44 ± 0.31 |
| Mammography CMMD | 80.72 ± 1.77 |
| Hateful Meme | 76.20 ± 1.85 |
| Justin Instagram | 95.92 ± 1.09 |
| Glaucoma SMDG | 89.09 ± 0.68 |
| Adoption | 47.04 ± 0.75 |
| CheXpert | 70.32 ± 0.62 |

### Regression · RMSE ↓

| Dataset | ViTab-Adapter |
| --- | ---: |
| Mango Mass | 0.0344 ± 0.0020 |
| Amazon Bestseller | 0.5721 ± 0.0091 |
| Pawpularity | 17.2254 ± 0.0236 |
| MkPhoto Bots | 0.0918 ± 0.0035 |
| Anime | 0.4213 ± 0.0048 |

## 7. Third-party code and attribution

The benchmark loading and preprocessing helpers under `mmpfn` are adapted from MultiModalPFN. See [LICENSE-MultiModalPFN](LICENSE-MultiModalPFN).

Built with PriorLabs-TabPFN (attribution for the reused benchmark helpers).
