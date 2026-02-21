# Nexus

**Config-driven Flux.2 Klein training on precomputed SSTK data.** A production-ready, YAML-based framework for LoRA and full fine-tuning of diffusion transformers (DiT) with validation, MLflow tracking, and checkpointing.

---

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Project structure](#project-structure)
- [Configuration system](#configuration-system)
- [Training](#training)
- [Dataset preparation](#dataset-preparation)
- [Loss functions](#loss-functions)
- [LoRA configuration](#lora-configuration)
- [Experiment tracking](#experiment-tracking)
- [Output and checkpoints](#output-and-checkpoints)
- [Examples and recipes](#examples-and-recipes)
- [Extending Nexus](#extending-nexus)
- [Troubleshooting](#troubleshooting)

---

## Overview

Nexus is a training framework for **Flux.2 Klein** and compatible diffusion transformer models. It focuses on:

- **Precomputed latents and text embeddings** — VAE and text encoder run once during dataset preparation, not during training. This significantly reduces GPU memory and speeds up training.
- **YAML-first configuration** — All hyperparameters, model choices, loss functions, and experiment settings live in config files. Configs can extend a base and override specific sections.
- **LoRA and full fine-tuning** — Train LoRA adapters (low-rank) or the full transformer. LoRA is the default and is memory-efficient.
- **Multiple data regimes** — SSTK (large-scale text-to-image), DreamBooth (instance + class), distillation (teacher-student).

---

## Features

| Feature | Description |
|---------|-------------|
| **Pipeline-based model loading** | Pipeline config defines the single source (`pretrained_model_name_or_path`). VAE, scheduler, and other components load from there. DiT is configured separately. |
| **Flow-matching loss** | MSE, L1, Huber, LogCosh with optional weighting schemes (sigma_sqrt, logit_normal, mode, cosmap). |
| **Prior preservation** | DreamBooth-style instance + class training with configurable prior weight. |
| **Distillation** | Flow loss + distillation from a frozen teacher. Teacher is loaded on-the-fly. |
| **Validation** | Periodic image generation and logging to TensorBoard / WandB / MLflow when `validation.prompt` is set. |
| **Accelerate** | Multi-GPU (DDP, FSDP), mixed precision (fp16, bf16), gradient checkpointing. |
| **Checkpointing** | Periodic saves, resume from checkpoint, configurable limit. |

---

## Architecture

### Model loading

```
pipeline.pretrained_model_name_or_path  ← single source
    ├── VAE        (for latent bn stats during training)
    ├── Scheduler  (flow-matching)
    ├── Text encoder, tokenizer  (inference only)
    └── DiT        ← loaded separately from model.dit
```

- **Training** loads: scheduler, DiT, and (briefly) VAE for latent batch-norm statistics.
- **Inference** loads everything from the pipeline via `pipeline.from_pretrained(...)`.

### Data flow

```
Images + captions  →  Prepare (MDS)  →  Precompute (latents + embeddings)  →  Training
```

Precompute runs VAE and text encoder once; training only reads precomputed latents and text embeddings.

---

## Quick start

### 1. Install

```bash
pip install -e .
```

### 2. Prepare data (or use existing MDS)

```bash
cd datasets/prepare/sstk
./run.sh all   # prepare + precompute
```

Or use an existing MDS directory with precomputed latents.

### 3. Train

```bash
./scripts/train.sh configs/klein4b/run1.yaml \
  --precomputed_data_dir /path/to/mds_latents \
  --output_dir ./runs/exp1
```

With more overrides:

```bash
./scripts/train.sh configs/klein4b/run1.yaml \
  --precomputed_data_dir /path/to/mds_latents \
  --output_dir ./runs/exp1 \
  --max_train_steps 2000 \
  --resume_from_checkpoint latest
```

---

## Installation

### From source (recommended)

```bash
git clone https://github.com/rchittersu/Nexus.git
cd Nexus
pip install -e .
```

### With conda

```bash
conda create -n nexus python=3.10
conda activate nexus
pip install -e .
```

### Verify setup

```bash
python check_setup.py
```

Checks dependencies (does not install); useful before training.

### As a dependency

```bash
# From Git
pip install git+https://github.com/rchittersu/Nexus.git

# Local path
pip install -e /path/to/Nexus
```

Then `from nexus.train import main` or `import nexus`.

---

## Prerequisites

| Requirement | Description |
|-------------|-------------|
| **Python** | 3.10+ |
| **Data** | MDS shards with precomputed VAE latents and text embeddings (see [Dataset preparation](#dataset-preparation)) |
| **Model** | `black-forest-labs/FLUX.2-klein-base-4B` from Hugging Face (or compatible pipeline) |
| **GPU** | Recommended for training. CPU possible but slow. |
| **Accelerate** | Run `accelerate config` before first use (multi-GPU, mixed precision). |

---

## Project structure

```
Nexus/
├── configs/klein4b/
│   ├── base.yaml           # Shared Flux2 Klein 4B defaults
│   ├── run1.yaml           # Experiment config (extends base)
│   ├── dreambooth.yaml     # DreamBooth instance + class
│   └── distillation.yaml   # Distillation training
├── scripts/
│   └── train.sh            # Training entrypoint
├── src/nexus/
│   ├── train/
│   │   ├── main.py         # Entrypoint, model load, training loop
│   │   ├── config.py       # YAML load, extends, class resolution
│   │   ├── train_loop.py   # Flow-matching step
│   │   ├── losses/         # FlowMatching, PriorPreservation, Distillation
│   │   └── validation.py  # Image gen + tracker logs
│   ├── models/
│   │   └── transformer_wrapper.py   # LoRA/full save/load
│   ├── data/
│   │   ├── precomputed_sstk_dataset.py
│   │   ├── precomputed_dreambooth_dataset.py
│   │   ├── t2i_dataset.py  # Streaming T2I (used by precompute)
│   │   └── utils.py        # Text preprocessing
│   └── utils/              # Device/dtype, MDS helpers
├── datasets/
│   ├── precompute.py       # MDS → latents + embeddings
│   ├── prepare/sstk/       # SSTK: images_txt + captions → MDS
│   └── prepare/dreambooth/ # DreamBooth: instance + class → MDS
├── tests/
├── check_setup.py
└── pyproject.toml
```

---

## Configuration system

### Inheritance

Configs can extend a base file via `extends`:

```yaml
# configs/klein4b/run2.yaml
extends: base.yaml

train:
  max_steps: 2000
  batch_size: 8
  learning_rate: 2.0e-4

validation:
  prompt: "a photo of a sks dog"
  steps: 250

output_dir: runs/run2
```

`extends` is resolved relative to the config file. The child is deep-merged over the base.

### Core sections

| Section | Purpose |
|---------|---------|
| `pipeline` | Top-level. `pretrained_model_name_or_path`, `class_name`. Single source for all components except DiT. |
| `model.dit` | DiT class and subfolder. Loaded separately from pipeline. |
| `model.transformer_wrapper` | LoRA component name (`transformer` for Flux2, `unet` for SD-style). |
| `dataset` | Dataset class + kwargs (`local`, `resolution`, `latent_channels`, `text_embed_hidden`, etc.). |
| `collate` | Collate function for the dataloader. |
| `train` | Batch size, epochs, max_steps, LR, accumulation, gradient checkpointing, etc. |
| `train_mode` | `lora` or `full`. |
| `lora` | Rank, alpha, dropout, `target_modules`. |
| `loss` | Loss class + kwargs (base, weighting_scheme, etc.). |
| `optimizer` | Optimizer class + kwargs. |
| `validation` | Steps, prompt (set to enable), num_images, seed. |
| `mlflow` | Experiment name, tracking URI. |

### Pipeline and DiT

```yaml
# Pipeline: load VAE, scheduler, text_encoder, etc. from here
pipeline:
  class_name: diffusers:Flux2KleinPipeline
  pretrained_model_name_or_path: black-forest-labs/FLUX.2-klein-base-4B

# DiT: loaded separately
model:
  dit:
    class_name: diffusers:Flux2Transformer2DModel
    subfolder: transformer
```

### Dataset kwargs

| Key | Default | Description |
|-----|---------|-------------|
| `local` | required | Path to MDS shards (precomputed latents). Overridden by `--precomputed_data_dir`. |
| `resolution` | 512 | Latent resolution. Must match precompute. |
| `latent_channels` | 32 | Flux.2 VAE latent channels. |
| `text_embed_hidden` | 7680 | Text embedding dim (2560×3 for Flux.2 Klein). 0 = auto-infer. |
| `shuffle` | true | Shuffle dataset. |

---

## Training

### Script (recommended)

```bash
./scripts/train.sh <config.yaml> [--output_dir ...] [--precomputed_data_dir ...] [--max_train_steps ...] ...
```

All extra arguments are passed to `accelerate launch -m nexus.train.main`.

### Direct invoke

```bash
accelerate launch -m nexus.train.main \
  --config configs/klein4b/run1.yaml \
  --precomputed_data_dir /path/to/mds \
  --output_dir ./out \
  --max_train_steps 1000 \
  --resume_from_checkpoint latest
```

### CLI overrides

| Flag | Effect |
|------|--------|
| `--config` | Required. Path to YAML config. |
| `--precomputed_data_dir` | Overrides `dataset.kwargs.local` |
| `--output_dir` | Overrides `output_dir` |
| `--max_train_steps` | Overrides `train.max_steps` |
| `--resume_from_checkpoint` | Path or `latest` |

### Multi-GPU

Configure once:

```bash
accelerate config
```

Select multi-GPU, mixed precision (fp16/bf16), etc. Then run as usual; Accelerate handles distribution.

### Reproducibility

- Config path is logged at startup.
- A copy is saved to `{output_dir}/config.yaml` for reproducibility.

---

## Dataset preparation

Training expects MDS shards with precomputed VAE latents and text embeddings. Two steps:

1. **Prepare** — images + captions → MDS (image, caption, width, height).
2. **Precompute** — MDS → VAE latents + text embeddings → MDS with `latents_512`, `text_embeds`, etc.

### SSTK (large-scale)

Images from a text file, captions from `<path>.json`.

```bash
cd datasets/prepare/sstk
./run.sh prepare   # images_txt → MDS
./run.sh precompute  # MDS → latents + embeddings
./run.sh all       # both
```

Edit `run.sh` for paths: `IMAGES_TXT`, `MDS_DIR`, `LATENTS_DIR`, or set `DATAROOT`.

**Prepare options**: `--images_txt`, `--local_mds_dir`, `--num_proc`, `--size`, `--min_size`, `--min_aspect_ratio`, `--max_aspect_ratio`.

### DreamBooth (instance + class)

Example: dog subject with prior preservation.

```bash
cd datasets/prepare/dreambooth
./run.sh prepare   # downloads diffusers/dog-example, writes MDS
./run.sh precompute
./run.sh all
```

**Prior preservation (class images):**

```bash
GENERATE_CLASS_IMAGES=1 ./run.sh prepare
```

Generates 100 class images with Flux2KleinPipeline when not present. Requires GPU.

### Precompute (shared)

`datasets/precompute.py` works with any prepare output. MDS must have columns: `image`, `caption`, `width`, `height`.

```bash
python datasets/precompute.py \
  --datadir ./mds/ \
  --savedir ./mds_latents_flux2/ \
  --num_proc 8 \
  --dataloader_workers 4 \
  --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-4B \
  --batch_size 32 \
  --resolution 512 \
  --model_dtype bfloat16 \
  --save_dtype float16
```

**Layout**: Flat (shards 0, 1, 2...) or nested (shards under subfolders). Each group is processed with up to `num_proc` workers.

**Output**: MDS with `caption`, `latents_512`, `text_embeds`. Use as `dataset.kwargs.local` or `--precomputed_data_dir`.

---

## Loss functions

Each loss receives `LossContext` and returns `(scalar, log_dict)`.

### FlowMatchingLoss (default)

Flow-matching with configurable base loss and weighting.

```yaml
loss:
  class_name: nexus.train.losses:FlowMatchingLoss
  weighting_scheme: none   # none | sigma_sqrt | logit_normal | mode | cosmap
  logit_mean: 0.0
  logit_std: 1.0
  mode_scale: 1.29
  kwargs:
    base: mse   # mse | l1 | huber | logcosh
    huber_delta: 1.0   # when base=huber
```

### FlowMatchingWithPriorPreservation (DreamBooth)

Instance + class prior preservation. Batch must be even (first half instance, second half class).

```yaml
loss:
  class_name: nexus.train.losses:FlowMatchingWithPriorPreservation
  kwargs:
    base: mse
    weight: 1.0   # prior loss weight
```

### DistillationLoss

Flow loss + distillation from a frozen teacher. Teacher is loaded automatically.

```yaml
loss:
  class_name: nexus.train.losses:DistillationLoss
  kwargs:
    base: mse
    pretrained_model_name_or_path: black-forest-labs/FLUX.2-klein-base-4B
    flow_weight: 0.5
    distillation_weight: 0.5
```

`transformer_cls`, `device`, `dtype` are injected at runtime from the model config.

---

## LoRA configuration

```yaml
train_mode: lora

lora:
  rank: 4
  alpha: 4
  dropout: 0.0
  target_modules: ["to_k", "to_q", "to_v", "to_out.0"]
```

- **rank**: LoRA rank (e.g. 4, 8, 16).
- **alpha**: Scaling. Often `alpha = rank`.
- **target_modules**: Linear layers to adapt. Flux2: `to_k`, `to_q`, `to_v`, `to_out.0`.

---

## Experiment tracking

With `report_to: mlflow` (default in Klein configs):

- Logs to `{output_dir}/mlruns`.
- Metrics: `loss`, `lr`; components as `loss/flow`, `loss/instance`, `loss/prior`, `loss/distillation`.
- Validation images when `validation.prompt` is set.
- Config and hyperparams stored with the run.

```bash
mlflow ui --backend-store-uri ./runs/exp1/mlruns
```

---

## Output and checkpoints

```
{output_dir}/
├── config.yaml           # Copy of config (reproducibility)
├── mlruns/               # MLflow (if report_to: mlflow)
├── logs/
└── checkpoint-{step}/    # Accelerate checkpoints
```

**Final weights:**

- **LoRA**: `transformer_lora.safetensors` in `output_dir`.
- **Full**: `transformer.safetensors` in `output_dir`.

`checkpoints_total_limit` in config controls how many checkpoints are retained.

---

## Examples and recipes

### New experiment from base

1. Copy a run config:
   ```bash
   cp configs/klein4b/run1.yaml configs/klein4b/run2.yaml
   ```
2. Edit `run2.yaml`:
   ```yaml
   extends: base.yaml
   train:
     max_steps: 2000
     learning_rate: 2.0e-4
   validation:
     prompt: "a photo of a sks dog"
   output_dir: runs/run2
   mlflow:
     experiment_name: flux2-run2
   ```
3. Run:
   ```bash
   ./scripts/train.sh configs/klein4b/run2.yaml --precomputed_data_dir /path/to/mds
   ```

### DreamBooth with prior preservation

```bash
cd datasets/prepare/dreambooth
GENERATE_CLASS_IMAGES=1 ./run.sh prepare
./run.sh precompute
cd ../..
./scripts/train.sh configs/klein4b/dreambooth.yaml --precomputed_data_dir ./dreambooth/mds_latents_flux2
```

### Distillation run

```bash
./scripts/train.sh configs/klein4b/distillation.yaml \
  --precomputed_data_dir /path/to/mds \
  --output_dir runs/distillation \
  --max_train_steps 1000
```

---

## Extending Nexus

### Adding a loss

1. Implement a callable that takes `LossContext` and returns `(scalar, log_dict)`.
2. Register in `nexus.train.losses` and reference via `loss.class_name`.
3. Add any required kwargs to the config.

### Supporting a new pipeline

1. Add a pipeline component mapping in `main.py` (`_PIPELINE_COMPONENTS`).
2. Define `vae` and `scheduler` (class, subfolder) for the pipeline class.
3. Add a `model.dit` config for the DiT.

### Adding a dataset

1. Implement a `StreamingDataset` (or compatible) that yields `latents`, `text_embeds`, `text_ids`.
2. Add a collate function.
3. Register in config via `dataset.class_name` and `collate.class_name`.

---

## Troubleshooting

### "pipeline.pretrained_model_name_or_path is required"

Ensure `pipeline` is defined at top level (or under `model`) with `pretrained_model_name_or_path`.

### "dataset.kwargs.local is required"

Set `dataset.kwargs.local` in config or pass `--precomputed_data_dir` on the command line.

### MPS (Apple Silicon) and bf16

`bf16` is not supported on MPS. Use `mixed_precision: fp16` or `null`.

### Out-of-memory

- Reduce `train.batch_size`.
- Enable `train.gradient_checkpointing: true`.
- Use LoRA instead of full fine-tuning.
- Lower resolution in precompute and dataset.

### Tests

```bash
pytest -v
```

Covers config, losses, data, and dataset preparation.

### Lint and format

```bash
./lint.sh          # check
./lint.sh --fix    # auto-fix
./lint.sh format   # format
```

---

## License

See the repository for license information.
