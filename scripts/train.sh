#!/usr/bin/env bash
# Launch training with given config. Forwards all extra args to accelerate.
#
# Usage:
#   ./scripts/train.sh configs/klein4b/t2i_finetune.yaml
#   ./scripts/train.sh configs/klein4b/t2i_finetune.yaml --precomputed_data_dir /path/to/mds
#   ./scripts/train.sh configs/klein4b/t2i_finetune.yaml --output_dir ./runs/exp1 --max_train_steps 2000

set -e
CONFIG="${1:?Usage: $0 <config.yaml> [--output_dir ...] [--precomputed_data_dir ...] ...}"
shift
accelerate launch -m nexus.train.main --config "$CONFIG" "$@"
