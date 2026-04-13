from __future__ import annotations

import argparse
import json

import torch
from PIL import Image

from src.data.transforms import DualViewTransform
from src.models.ladl_net import LADLNet
from src.utils.io import load_checkpoint, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LADL-Net inference on a single frame.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--bbox", nargs=4, type=float, default=None)
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    state = load_checkpoint(args.checkpoint, map_location="cpu")
    classes = state.get("class_names") or cfg["dataset"].get("classes")
    if not classes:
        raise ValueError("Inference requires explicit `dataset.classes` in the config.")

    image = Image.open(args.image).convert("RGB")
    transform = DualViewTransform(cfg["transforms"], split="test")
    sample = transform(image=image, bbox=args.bbox)
    batch = {
        "global_image": sample["global_image"].unsqueeze(0),
        "local_image": sample["local_image"].unsqueeze(0),
        "attention_prior": sample["attention_prior"].unsqueeze(0),
        "has_bbox": torch.tensor([sample["has_bbox"]], dtype=torch.float32),
    }

    model = LADLNet(num_classes=len(classes), class_names=classes, model_cfg=cfg["model"], loss_cfg=cfg["loss"])
    model.load_state_dict(state["model"], strict=False)
    device = torch.device(cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.inference_mode():
        outputs = model(batch)
        probs = outputs["logits"].softmax(dim=1)[0]
        top_probs, top_idx = probs.topk(min(args.topk, len(classes)))
    result = [{"class": classes[i.item()], "probability": round(p.item(), 6)} for p, i in zip(top_probs, top_idx)]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
