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
accelerate launch -m nexus.train.main \
  --config configs/klein4b/run1.yaml \
  --precomputed_data_dir /path/to/mds_latents \
  --output_dir ./runs/exp1
```

Or use `./scripts/train.sh` as a wrapper.

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
# run2.yaml
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

**Main sections:** `pipeline`, `model.dit`, `dataset`, `train`, `train_mode`, `lora`, `loss`, `optimizer`, `validation`, `mlflow`

---

## Training

| CLI | Effect |
|-----|--------|
| `--config` | Required. YAML path. |
| `--precomputed_data_dir` | Overrides `dataset.kwargs.local` |
| `--output_dir` | Overrides `output_dir` |
| `--max_train_steps` | Overrides `train.max_steps` |
| `--resume_from_checkpoint` | Path or `latest` |

---

## Dataset preparation

1. **Prepare** — images + captions → MDS
2. **Precompute** — MDS → VAE latents + text embeddings → MDS with `latents_512`, `text_embeds`

**SSTK:**
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

```
{output_dir}/
├── config.yaml
├── mlruns/           # MLflow (report_to: mlflow)
├── logs/
├── checkpoint-{step}/
└── transformer_lora.safetensors  # or transformer.safetensors (full)
```

`mlflow ui --backend-store-uri ./runs/exp1/mlruns`

---

## Requirements

- Python 3.10+
- MDS shards with precomputed latents (from prepare + precompute)
- `black-forest-labs/FLUX.2-klein-base-4B` or compatible
- `accelerate config` before first run

---

## Project layout

```
configs/klein4b/     # base, run1, dreambooth, distillation
scripts/train.sh     # wrapper for accelerate launch
src/nexus/
├── train/           # main, config, train_loop, losses, validation
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
| MPS + bf16 | Use `fp16` or `null` (bf16 not supported on Apple Silicon) |
| OOM | Lower batch_size, enable gradient_checkpointing, use LoRA |

```bash
pytest -v
```
