#!/usr/bin/env bash
# Runs training with the specified config.
# Extra arguments are passed through, so a short run is:
#   scripts/train.sh --episodes 10
set -e
python src/main.py --config configs/train.yaml "$@"
