# LADL-Net: Lesion-Aware Dual-View Long-Tail Network

LADL-Net is a research-grade PyTorch project for labeled-only long-tailed capsule endoscopy classification on Kvasir-Capsule. The method treats lesion bounding boxes as a first-class signal by combining a global frame branch, a bbox-guided local crop branch, differentiable bbox-prior spatial attention, gated dual-view fusion, Balanced Softmax, supervised contrastive learning, and cross-view consistency.

## Method Summary

For each labeled frame, LADL-Net constructs:

1. A global view from the full frame.
2. A local view from a bbox-expanded crop.

The shared encoder extracts global and local representations. A differentiable soft bbox prior is resized and injected into the global feature map before pooling. The two features are fused by a learnable gate:

`alpha = sigmoid(W([f_global; f_local]))`

`f_fused = alpha * f_local + (1 - alpha) * f_global`

The fused embedding drives the main classifier while auxiliary heads, supervised contrastive loss, attention alignment, and view consistency improve robustness under severe class imbalance.

## Dataset Assumptions

Expected metadata columns:

- `filename`
- `label`
- `video_id`
- `frame_number`
- bbox columns such as `x1,y1,x2,y2` or `bbox_x,bbox_y,bbox_w,bbox_h`

Recommended layout:

```text
dataset_root/
├── metadata.csv
├── images/...
└── ...
```

If `metadata.csv` is missing, the code falls back to scanning class folders and infers `label`, `video_id`, and `frame_number` from the directory and filename. Missing or invalid boxes are handled gracefully.

## Training

```bash
python train.py --config configs/default.yaml --build-splits
```

## Evaluation

```bash
python test.py --config configs/default.yaml --checkpoint outputs/ladl_default/checkpoints/best_macro_f1.pt
```

## Inference

```bash
python infer.py \
  --config configs/default.yaml \
  --checkpoint outputs/ladl_default/checkpoints/best_macro_f1.pt \
  --image path/to/frame.jpg \
  --bbox 32 40 180 200
```

## Outputs

Each run writes:

- checkpoints by macro F1 and MCC
- predictions and logits CSV
- per-class tables
- confusion matrices
- bootstrap confidence intervals
- ROC and PR plots
- embedding visualizations
- training curves

## Suggested Ablations

1. Full-frame only vs local-only vs dual-view.
2. Dual-view without attention vs with bbox attention.
3. `ce` vs `weighted_ce` vs `focal` vs `balanced_softmax`.
4. With and without supervised contrastive loss.
5. DINOv2 vs MambaVision-compatible backbones.
6. Crop expansion ratio sweep from `1.2` to `1.8`.

The implementation is strictly labeled-only and avoids unlabeled frames, pseudo-labeling, SSL pretraining on Kvasir-Capsule, or external VCE datasets.
