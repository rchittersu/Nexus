# Nexus

Personal DiT (Diffusion Transformer) training stack — similar in spirit to [diffusers](https://github.com/huggingface/diffusers), but tailored to your own workflows.

## Layout

- **`src/nexus/`** — Core **definitions** (library-style):
  - `archs` — DiT blocks (`DiTBlockBase`), embedders (`EmbedderBase`), etc.
  - `pipelines` — diffusion pipelines, schedulers, sampling (placeholders)
  - `train` — training loops, logging, checkpointing, config
  - `utils` — shared helpers (device, dtype via `DATA_TYPES`, …)
  - `data` — dataset interfaces (`StreamingT2IDataset`), text preprocessing, transforms

- **`datasets/prepare/sstk/`** — SSTK-style dataset preparation:
  - `prepare.py` — images → MDS shards (filtering by size/aspect ratio)
  - `precompute.py` — MDS shards → precomputed latents + text embeddings (VAE, text encoder)
  - `run.sh` — orchestration script for prepare + precompute

- **Project root** — Extended implementations and entrypoints:
  - `run_train.py` — example training entrypoint; import from `nexus` and run your loop

## Install

```bash
pip install -e .
# or with training deps (torch, diffusers, transformers):
pip install -e ".[train]"
# with dev tools (pytest, ruff):
pip install -e ".[train,dev]"
```

## Usage

From the repo root:

```python
from nexus.archs import DiTBlockBase, EmbedderBase
from nexus.data.t2i_dataset import StreamingT2IDataset
from nexus.utils import DATA_TYPES

# Your training entrypoint would use these core definitions
```

## Dataset preparation (SSTK)

For SSTK-style pipelines (e.g. SA1B), images are first packed into MDS shards, then latents and embeddings are precomputed with a pretrained VAE + text encoder (e.g. FLUX.2-klein):

```bash
cd datasets/prepare/sstk
./run.sh [prepare|precompute|all]
```

Configure via env vars (see `run.sh`) or edit the script:
- `IMAGES_TXT`, `LOCAL_MDS_DIR`, `SIZE`, `PRETRAINED_MODEL`, etc.
- Output goes to `SAVEDIR` (default `./sa1b/mds_latents_flux2/`).

Keep **definitions** in `src/nexus/`; keep **concrete runs and extensions** at the root or in sibling folders.
