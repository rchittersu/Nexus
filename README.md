# Nexus

Config-driven Flux.2 Klein training on precomputed SSTK data. YAML-based configs, LoRA or full fine-tuning, validation, MLflow tracking, and checkpointing.

---

## Quick start

```bash
# Install (from repo root)
pip install -e .

# Train (config is the only required arg)
./scripts/train.sh configs/klein4b/run1.yaml
```

With overrides:

```bash
./scripts/train.sh configs/klein4b/run1.yaml \
  --precomputed_data_dir /path/to/mds_latents \
  --output_dir ./runs/exp1 \
  --max_train_steps 2000
```

---

## Prerequisites

- **Data**: MDS shards with precomputed VAE latents and text embeddings (see [Dataset preparation](#dataset-preparation))
- **Model**: `black-forest-labs/FLUX.2-klein-base-4B` (or compatible) from Hugging Face
- **GPU**: Recommended for training

---

## Install

```bash
pip install -e .
```

Verify setup (checks only; does not install):

```bash
python check_setup.py
```

Use as a dependency in another project:

```bash
# From Git
pip install git+https://github.com/your-org/Nexus.git

# Or from a local path
pip install -e /path/to/Nexus
```

Then `from nexus.train import main` or `import nexus`.

## Testing

```bash
pip install -e .
pytest -v
```

Lint and format with ruff:

```bash
./lint.sh          # check only
./lint.sh --fix    # auto-fix
./lint.sh format   # format code
```

Unit tests cover config (load, extends, ns_to_kwargs), losses (MSE, L1, Huber, LogCosh), data (collate), and dataset prep (prepare, precompute).

Accelerate config:

```bash
accelerate config
```

Choose multi-GPU / single-GPU and mixed precision as needed.

---

## Project structure

```
Nexus/
├── configs/
│   └── klein4b/
│       ├── base.yaml        # Shared Flux2 Klein 4B defaults
│       ├── run1.yaml        # Experiment config (extends base)
│       └── distillation.yaml  # Distillation training (teacher-student)
├── scripts/
│   └── train.sh           # Training entrypoint
├── check_setup.py         # Verify deps (python check_setup.py)
├── lint.sh                # Ruff lint/format (./lint.sh)
├── src/nexus/
│   ├── train/             # Training
│   │   ├── main.py        # Entrypoint
│   │   ├── config.py      # YAML load, extends, class resolution
│   │   ├── train_loop.py  # Flow-matching step
│   │   ├── losses.py      # MSE, L1, Huber, LogCosh, DistillationLoss
│   │   └── validation.py  # Image gen + tracker logs
│   ├── models/
│   │   └── transformer_wrapper.py   # Generic LoRA/full wrapper
│   ├── data/
│   │   ├── precomputed_sstk.py  # MDS dataset
│   │   ├── t2i_dataset.py       # Streaming T2I (used by precompute)
│   │   └── utils.py            # Text preprocessing for captions
│   └── utils/              # Device/dtype helpers
└── datasets/prepare/sstk/  # Data prep scripts
    ├── base.py             # Streaming dataloader builder
    ├── prepare.py          # Images → MDS
    ├── precompute.py       # MDS → latents + embeddings
    └── run.sh
```

---

## Training

### Script (recommended)

```bash
./scripts/train.sh <config.yaml> [accelerate args...]
```

Config is required; other args are passed through to `accelerate launch -m nexus.train.main`.

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

### Logging and reproducibility

- Config path is logged at startup.
- A copy of the config is saved to `{output_dir}/config.yaml` for reproducibility.

---

## Dataset preparation

Training expects MDS shards with precomputed latents and text embeddings. Two steps:

### 1. Prepare — images → MDS

Packs images and captions into MDS shards.

```bash
cd datasets/prepare/sstk
./run.sh prepare
```

Edit `run.sh` to change paths and parameters (e.g. `--images_txt`, `--local_mds_dir`, `--size`, `--datadir`, `--savedir`).

### 2. Precompute — MDS → latents + embeddings

Encodes images with the VAE and text with the encoder; writes new MDS shards.

```bash
./run.sh precompute
```

Or both:

```bash
./run.sh all
```

Output: MDS directory with `latents_512`, `text_embeds`, `text_ids`, etc. Use this path as `dataset.kwargs.local` or `--precomputed_data_dir`.

---

## Config system

### Inheritance

Configs can extend a base file:

```yaml
# configs/klein4b/run1.yaml
extends: base.yaml

train:
  max_steps: 1000
  batch_size: 8
output_dir: runs/exp1
```

`extends` is resolved relative to the config file. The child config is deep-merged over the base.

### Key sections

| Section | Purpose |
|---------|---------|
| `dataset` | Class + kwargs for `PrecomputedSSTKDataset` |
| `model` | Transformer, VAE, scheduler, pipeline, wrapper (Flux2 Klein by default) |
| `train` | Batch size, steps, LR, gradient accumulation, etc. |
| `loss` | `class_name` (MSELoss, L1Loss, HuberLoss, LogCoshLoss, DistillationLoss, MetaLoss) + kwargs |
| `distillation` | `source_transformer` path + `alpha` for teacher-student training |
| `lora` | Rank, alpha, dropout, target_modules |
| `validation` | Steps, prompt (set to enable), num_images |
| `mlflow` | `experiment_name`, `tracking_uri` |

### Loss

```yaml
loss:
  class_name: nexus.train.losses:MSELoss   # default
  # nexus.train.losses:L1Loss
  # nexus.train.losses:HuberLoss  # add kwargs: { delta: 1.0 }
  # nexus.train.losses:LogCoshLoss
  weighting_scheme: none  # none, sigma_sqrt, logit_normal, mode, cosmap
```

### Distillation

Train with a source (teacher) transformer. `DistillationLoss` is pure distillation (pred vs teacher_pred only). Use `MetaLoss` to combine flow-matching and distillation:

```yaml
distillation:
  source_transformer:
    pretrained_model_name_or_path: path/to/teacher
    class_name: diffusers:Flux2Transformer2DModel
    subfolder: transformer

loss:
  class_name: nexus.train.losses:MetaLoss
  kwargs:
    losses:
      - class_name: nexus.train.losses:MSELoss
        scale: 0.5
        name: flow
      - class_name: nexus.train.losses:DistillationLoss
        scale: 0.5
        name: distillation
```

### MetaLoss

Combine multiple losses with configurable scales: `L = sum(scale_i * loss_i(...))`. Each component is logged as `loss/{name}` (e.g. in TensorBoard/MLflow):

```yaml
loss:
  class_name: nexus.train.losses:MetaLoss
  kwargs:
    losses:
      - class_name: nexus.train.losses:MSELoss
        scale: 1.0
        name: flow
      - class_name: nexus.train.losses:L1Loss
        scale: 0.1
        name: l1
```

When distillation is enabled, the loss must include `DistillationLoss` (e.g. via MetaLoss). If `DistillationLoss` is used without distillation config, training will raise an error.

### Transformer wrapper

Generic for Flux2, SD3, etc. Use `component_name: transformer` (Flux2) or `component_name: unet` (SD-style).

---

## Experiment tracking (MLflow)

With `report_to: mlflow` (default in Klein configs):

- Logs go to `{output_dir}/mlruns`
- Metrics: loss, lr; MetaLoss components logged as loss/flow, loss/distillation, etc.
- Validation images: logged when `validation.prompt` is set
- Config and hyperparams: stored with the run

View locally:

```bash
mlflow ui --backend-store-uri ./runs/exp1/mlruns
```

---

## Output layout

```
{output_dir}/
├── config.yaml           # Copy of config used (reproducibility)
├── mlruns/               # MLflow tracking (if report_to: mlflow)
├── logs/                 # Additional logs
└── checkpoint-{step}/    # Checkpoints
```

Final weights:

- **LoRA**: `transformer_lora.safetensors` in output_dir
- **Full**: `transformer.safetensors` in output_dir

---

## Example: new experiment

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
