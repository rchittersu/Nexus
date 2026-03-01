# Nexus

Config-driven Flux.2 Klein training on precomputed latents and text embeddings. YAML configs, LoRA or full fine-tune, MLflow tracking.

---

## Quick start

```bash
pip install -e .
accelerate config   # once: multi-GPU, precision, etc.
```

**Prepare data** (or use existing MDS):

```bash
cd datasets/prepare/sstk && ./run.sh all
```

**Train**:

```bash
./scripts/train.sh configs/klein4b/t2i_finetune.yaml \
  --precomputed_data_dir /path/to/mds_latents \
  --output_dir my-run
```

Or run `accelerate launch` directly (uses your default accelerate config; DDP by default).

---

## How it works

- **Pipeline** → VAE, scheduler, text encoder from `pretrained_model_name_or_path`
- **DiT** → loaded separately via `model.dit`
- **Data** → images + captions → Prepare (MDS) → Precompute (latents + embeddings) → training only reads precomputed

Training does not run VAE or text encoder; that happens once during precompute.

---

## Config

Configs extend a base and override sections:

```yaml
extends: base.yaml

train:
  max_steps: 2000
  batch_size: 8
  learning_rate: 2.0e-4

validation:
  entries:
    - prompt: "a photo of a sks dog"
      source: null
      num_images: 2
  steps: 250

mlflow:
  experiment_name: klein4b-t2i-finetune
  run_name: my-run
  user: null
```

**Main sections:** `pipeline`, `model.dit`, `dataset`, `train`, `train_mode`, `lora`, `loss`, `optimizer`, `validation`, `mlflow`

**mlflow required:** `experiment_name`, `run_name`. Output path: `log_root/experiments/{user}/{experiment_name}-{run_name}` (user from `mlflow.user`, defaults to `default` if null). MLflow run name is `{run_name}-{user}` for 1-to-1 mapping with output_dir; resume finds the run by name (no run_id file).

**validation** (inherited from base): `inference_steps: 4`, `guidance_scale: 1.0`. Override `steps`, `entries`, `resolution` per config. Each entry: `{prompt, source, num_images}` (source: null for t2i, path for img2img).

---

## Training

| train.sh / CLI | Effect |
|----------------|--------|
| `--config`, `-c` | Required. YAML path. |
| `--fsdp`, `-f` | Use FSDP (configs/accelerate_fsdp.yaml). Omit for default DDP. |
| `--cuda_visible_devices`, `-g` | GPU IDs (e.g. 0,1). train.sh only. |
| `--precomputed_data_dir` | Overrides `dataset.kwargs.local` |
| `--output_dir` | Overrides `mlflow.run_name` (run identifier in output path) |
| `--max_train_steps` | Overrides `train.max_steps` |
| `--auto_resume` | When output_dir has checkpoints, resume from latest and continue the same MLflow run |

**Existing output_dir:** If checkpoints exist and `--auto_resume` is not set, training errors. If no checkpoints exist, a warning is logged and a fresh run starts (new MLflow run).

**FSDP (Fully Sharded Data Parallel):** For memory-efficient multi-GPU training, pass `--fsdp`:

```bash
./scripts/train.sh configs/klein4b/t2i_finetune.yaml --fsdp \
  --precomputed_data_dir /path/to/mds_latents \
  --output_dir my-fsdp-run
```

LoRA + FSDP uses PEFT's wrap policy automatically. Adjust `num_processes` in `configs/accelerate_fsdp.yaml` for your GPU count. Without `--fsdp`, training uses your default accelerate config (typically DDP).

---

## Dataset preparation

1. **Prepare** — images + captions → MDS
2. **Precompute** — MDS → VAE latents + text embeddings → MDS with `latents_512`, `text_embeds`

**Standard (MDS):**
```bash
cd datasets/prepare/sstk
./run.sh prepare    # images_txt + captions → MDS
./run.sh precompute # MDS → latents + embeddings
./run.sh all        # both
```

**DreamBooth:**
```bash
cd datasets/prepare/dreambooth
GENERATE_CLASS_IMAGES=1 ./run.sh prepare   # optional: generate class images
./run.sh precompute
```

**Precompute** (shared script):
```bash
python datasets/precompute.py \
  --datadir ./mds/ \
  --savedir ./mds_latents/ \
  --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-4B \
  --num_proc 8 --resolution 512
```

---

## Losses

| Loss | Use case |
|------|----------|
| `FlowMatchingLoss` | Default. MSE/L1/Huber/LogCosh, optional weighting. |
| `FlowMatchingWithPriorPreservation` | DreamBooth instance + class |
| `DistillationLoss` | Flow + distillation from frozen teacher |

Config example:

```yaml
loss:
  class_name: nexus.losses:FlowMatchingLoss
  weighting_scheme: none
  kwargs:
    base: mse
```

---

## Output

**Project layout** (default `log_root: logs`):

```
logs/
├── mlruns/                                    # MLflow tracking store (project-level)
└── experiments/
    └── {user}/                                 # from mlflow.user, or "default"
        └── {experiment_name}-{run_name}/       # e.g. klein4b-t2i-finetune-batch4-lora4
        ├── config.yaml
        ├── checkpoint-{step}/
        ├── validation_images/
        └── transformer_lora.safetensors      # final save
```

**View MLflow:**

```bash
mlflow ui --backend-store-uri ./logs/mlruns --host 0.0.0.0
```

---

## Requirements

- Python 3.10+
- MDS shards with precomputed latents (from prepare + precompute)
- `black-forest-labs/FLUX.2-klein-base-4B` or compatible
- `accelerate config` before first run

---

## Project layout

```
configs/
├── accelerate_fsdp.yaml  # FSDP config for multi-GPU training
├── klein4b/               # base, t2i_finetune, t2i_dreambooth, t2i_distillation
└── klein4b-base/          # same structure, FLUX.2-klein-base-4B model
scripts/train.sh     # wrapper for accelerate launch; use --fsdp for FSDP
src/nexus/
├── train/           # main, config, train_loop, validation
├── losses/          # flow_matching, distillation, prior_preservation
├── data/            # precomputed datasets, collate
└── utils/           # checkpoint, log, train utils
datasets/
├── precompute.py
└── prepare/sstk, prepare/dreambooth
```

---

## Extending

- **New loss** — callable `(LossContext) -> (scalar, log_dict)`, register via `loss.class_name`
- **New pipeline** — add to `_PIPELINE_COMPONENTS` in `main.py`, define vae/scheduler + `model.dit`
- **New dataset** — yields `latents`, `text_embeds`, `text_ids`, collate, config `dataset.class_name`

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `dataset.kwargs.local is required` | Set in config or pass `--precomputed_data_dir` |
| `mlflow config required` | Config must have `mlflow.experiment_name` and `mlflow.run_name` |
| MPS + bf16 | Use `fp16` or `null` (bf16 not supported on Apple Silicon) |
| OOM | Lower batch_size, enable gradient_checkpointing, use LoRA |

```bash
pytest -v
```
