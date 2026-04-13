#!/usr/bin/env bash
set -euo pipefail

python train.py --config configs/ablation_dual_view.yaml --build-splits
python train.py --config configs/ablation_loss.yaml --build-splits
python train.py --config configs/ablation_backbone.yaml --build-splits
