#!/usr/bin/env bash
#
# Run img2img dataset preparation: prepare (pairs -> MDS) then precompute (MDS -> latents).
# Set DATAROOT or edit vars below to change data paths.
#
# Usage: ./run.sh [prepare|precompute|all]
#

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATAROOT="${DATAROOT:-./img2img}"
PAIRS_JSON="${DATAROOT}/pairs.json"
MDS_DIR="${DATAROOT}/mds"
LATENTS_DIR="${DATAROOT}/mds_latents_flux2"

run_prepare() {
    echo "=== Running img2img prepare ==="
    python "$SCRIPT_DIR/prepare.py" \
        --pairs_json "$PAIRS_JSON" \
        --local_mds_dir "${MDS_DIR}/" \
        --num_proc 4 \
        --seed 42 \
        --min_size 512
}

run_precompute() {
    echo "=== Running precompute (img2img mode) ==="
    # Requires precompute.py --mode img2img support (extends existing precompute)
    python "$DATASETS_ROOT/precompute.py" \
        --datadir "${MDS_DIR}/" \
        --savedir "${LATENTS_DIR}/" \
        --num_proc 4 \
        --resolution 512 \
        --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-4B \
        --batch_size 16 \
        --seed 42 \
        --model_dtype bfloat16 \
        --save_dtype float16 \
        --dataloader_workers 2 \
        --mode img2img
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
