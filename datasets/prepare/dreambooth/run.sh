#!/usr/bin/env bash
#
# Run DreamBooth dog example: prepare (dog-example -> MDS) then precompute (MDS -> latents).
#
# Usage: ./run.sh [prepare|precompute|all]
#

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATAROOT="${DATAROOT:-./dreambooth}"
DOG_DIR="${DATAROOT}/dog"
MDS_DIR="${DATAROOT}/mds"
LATENTS_DIR="${DATAROOT}/mds_latents_flux2"

run_prepare() {
    echo "=== Running DreamBooth prepare ==="
    python "$SCRIPT_DIR/prepare.py" \
        --dataset_name diffusers/dog-example \
        --download_dir "$DOG_DIR" \
        --instance_prompt "a photo of sks dog" \
        --local_mds_dir "$MDS_DIR" \
        --num_proc 1 \
        --min_size 256
}

run_precompute() {
    echo "=== Running precompute ==="
    python "$DATASETS_ROOT/precompute.py" \
        --datadir "$MDS_DIR" \
        --savedir "$LATENTS_DIR" \
        --num_proc 1 \
        --image_resolutions 512 \
        --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-4B \
        --batch_size 4 \
        --seed 42 \
        --model_dtype bfloat16 \
        --save_dtype float16 \
        --dataloader_workers 0
}

case "${1:-all}" in
    prepare)
        run_prepare
        ;;
    precompute)
        run_precompute
        ;;
    all)
        run_prepare
        run_precompute
        ;;
    *)
        echo "Usage: $0 [prepare|precompute|all]"
        exit 1
        ;;
esac
