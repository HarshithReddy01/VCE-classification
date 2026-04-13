from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import KvasirCapsuleDataset
from src.data.split_builder import load_or_build_splits
from src.engine.evaluator import Evaluator
from src.models.ladl_net import LADLNet
from src.utils.io import ensure_dir, load_checkpoint, load_config, merge_cli_overrides
from src.utils.logger import setup_logger
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LADL-Net.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, {"runtime.device": args.device, "runtime.output_dir": args.output_dir})
    output_dir = Path(cfg["runtime"]["output_dir"])
    ensure_dir(output_dir)
    logger = setup_logger(output_dir / f"eval_{args.split}.log")
    seed_everything(cfg["runtime"]["seed"], deterministic=cfg["runtime"].get("deterministic", True))
    split_paths = load_or_build_splits(cfg, fold=args.fold)
    state = load_checkpoint(args.checkpoint, map_location="cpu")

    class_names = state.get("class_names")
    if not class_names:
        train_ref = KvasirCapsuleDataset(
            dataset_root=cfg["dataset"]["root"],
            metadata_path=cfg["dataset"].get("metadata_path"),
            split_csv=split_paths["train"],
            classes=cfg["dataset"].get("classes"),
            use_filtered_classes=cfg["dataset"].get("use_filtered_classes", False),
            transform_cfg=cfg["transforms"],
            split="train",
        )
        class_names = train_ref.classes

    dataset = KvasirCapsuleDataset(
        dataset_root=cfg["dataset"]["root"],
        metadata_path=cfg["dataset"].get("metadata_path"),
        split_csv=split_paths[args.split],
        classes=class_names,
        use_filtered_classes=cfg["dataset"].get("use_filtered_classes", False),
        transform_cfg=cfg["transforms"],
        split=args.split,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["loader"].get("eval_batch_size", cfg["loader"]["batch_size"]),
        shuffle=False,
        num_workers=cfg["loader"]["num_workers"],
        pin_memory=cfg["loader"]["pin_memory"],
    )
    model = LADLNet(
        num_classes=len(class_names),
        class_names=class_names,
        model_cfg=cfg["model"],
        loss_cfg=cfg["loss"],
    )
    model.load_state_dict(state["model"], strict=False)
    device = torch.device(cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    model.to(device)

    evaluator = Evaluator(
        class_names=dataset.classes,
        output_dir=output_dir / f"{args.split}_evaluation",
        logger=logger,
        metrics_cfg=cfg["metrics"],
    )
    evaluator.evaluate(model=model, loader=loader, device=device, split_name=args.split)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
