#!/usr/bin/env bash
set -euo pipefail

CKPT=${1:-outputs/ladl_default/checkpoints/best_macro_f1.pt}
python test.py --config configs/default.yaml --checkpoint "$CKPT"
