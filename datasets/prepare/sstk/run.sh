#!/usr/bin/env bash
#
# Run SSTK dataset preparation: prepare.py (images -> MDS) then precompute.py (MDS -> latents).
#
# Usage: ./run.sh [prepare|precompute|all]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

run_prepare() {
    echo "=== Running prepare.py ==="
    cd "$PROJECT_ROOT"
    python "$SCRIPT_DIR/prepare.py" \
        --images_txt ./sa1b/image_paths.txt \
        --local_mds_dir ./sa1b/mds/ \
        --num_proc 16 \
        --seed 42 \
        --size 100000 \
        --min_size 512 \
        --min_aspect_ratio 0.67 \
        --max_aspect_ratio 1.33
}

run_precompute() {
    echo "=== Running precompute.py ==="
    cd "$PROJECT_ROOT"
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    accelerate launch --multi_gpu --num_processes 8 \
        -m datasets.prepare.sstk.precompute \
        --datadir ./sa1b/mds/ \
        --savedir ./sa1b/mds_latents_flux2/ \
        --image_resolutions 512 1024 \
        --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-4B \
        --batch_size 32 \
        --seed 42 \
        --model_dtype bfloat16 \
        --save_dtype float16
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
