# SSTK Dataset Preparation Tests

Tests for `prepare.py` (images → MDS) and `precompute.py` (MDS → latents + text embeddings).

## Requirements

- **Unit tests**: `pytest`, `numpy`, `PIL`, `mosaicml-streaming`
- **Integration tests**: above + `torch`, `diffusers`, `transformers`, `accelerate`; CUDA GPU for precompute

## Running Tests

From the **project root** (`Nexus/`):

```bash
# All unit tests (prepare + precompute)
python -m pytest datasets/prepare/sstk/tests/ -v -m "not integration"

# Prepare tests only
python -m pytest datasets/prepare/sstk/tests/test_prepare.py -v -m "not integration"

# Precompute unit tests only
python -m pytest datasets/prepare/sstk/tests/test_precompute.py -v -m "not integration"
```

### Integration Tests

Integration tests are slower and require additional setup:

```bash
# Prepare integration: 10 images → MDS → read → save PNGs with caption as filename
python -m pytest datasets/prepare/sstk/tests/test_prepare.py::TestIntegrationPrepareAndRead -v

# Precompute integration: prepare → precompute → read shards → decode latent + re-encode text → verify similarity
# Requires: CUDA, accelerate, FLUX.2-klein-base-4B model (cached)
python -m pytest datasets/prepare/sstk/tests/test_precompute.py::TestPrecomputeIntegration -v
```

## Test Files

| File | Coverage |
|------|----------|
| `test_prepare.py` | `parse_arguments`, `write_images` filtering (min_size, aspect_ratio, extensions), main orchestration, full prepare→read→save integration |
| `test_precompute.py` | `parse_args`, columns-building logic, full prepare→precompute→decode→verify integration |
