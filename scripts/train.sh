#!/usr/bin/env bash
# Launch training with given config.
#
# Usage:
#   ./scripts/train.sh configs/klein4b/t2i_finetune.yaml
#   ./scripts/train.sh --config configs/klein4b/t2i_finetune.yaml --precomputed_data_dir /path/to/mds
#   ./scripts/train.sh -c configs/klein4b/t2i_finetune.yaml --fsdp -g 0,1
#
# Flags:
#   --config, -c     Training config (required)
#   --fsdp, -f       Use FSDP config (configs/accelerate_fsdp.yaml); default is accelerate config
#   --cuda_visible_devices, -g   GPU IDs (e.g. 0,1)

set -e
CONFIG=""
CUDA_VISIBLE_DEVICES=""
FSDP=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config|-c)
            CONFIG="$2"
            shift 2
            ;;
        --fsdp|-f)
            FSDP=1
            shift
            ;;
        --cuda_visible_devices|-g)
            CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        *)
            if [[ -z "$CONFIG" && "$1" != --* ]]; then
                CONFIG="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

[[ -n "$CONFIG" ]] || { echo "Usage: $0 <config.yaml> | --config <config.yaml> [--fsdp] [--cuda_visible_devices 0,1] [--precomputed_data_dir ...] [--output_dir ...] ..."; exit 1; }

[[ -z "$CUDA_VISIBLE_DEVICES" ]] || export CUDA_VISIBLE_DEVICES

ACCELERATE_ARGS=()
[[ -n "$FSDP" ]] && ACCELERATE_ARGS+=(--config_file configs/accelerate_fsdp.yaml)

accelerate launch "${ACCELERATE_ARGS[@]}" -m nexus.train --config "$CONFIG" "${EXTRA_ARGS[@]}"
