from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import KvasirCapsuleDataset
from src.data.split_builder import SplitBuilder, load_or_build_splits
from src.engine.trainer import Trainer
from src.models.ladl_net import LADLNet
from src.utils.io import ensure_dir, load_config, merge_cli_overrides
from src.utils.logger import setup_logger
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LADL-Net on Kvasir-Capsule.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--build-splits", action="store_true")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def build_dataloaders(cfg: dict, split_paths: dict[str, Path]) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_cfg = cfg["dataset"]
    loader_cfg = cfg["loader"]
    train_set = KvasirCapsuleDataset(
        dataset_root=dataset_cfg["root"],
        metadata_path=dataset_cfg.get("metadata_path"),
        split_csv=split_paths["train"],
        classes=dataset_cfg.get("classes"),
        use_filtered_classes=dataset_cfg.get("use_filtered_classes", False),
        transform_cfg=cfg["transforms"],
        split="train",
    )
    val_set = KvasirCapsuleDataset(
        dataset_root=dataset_cfg["root"],
        metadata_path=dataset_cfg.get("metadata_path"),
        split_csv=split_paths["val"],
        classes=train_set.classes,
        use_filtered_classes=dataset_cfg.get("use_filtered_classes", False),
        transform_cfg=cfg["transforms"],
        split="val",
    )
    test_set = KvasirCapsuleDataset(
        dataset_root=dataset_cfg["root"],
        metadata_path=dataset_cfg.get("metadata_path"),
        split_csv=split_paths["test"],
        classes=train_set.classes,
        use_filtered_classes=dataset_cfg.get("use_filtered_classes", False),
        transform_cfg=cfg["transforms"],
        split="test",
    )
    common = {"num_workers": loader_cfg["num_workers"], "pin_memory": loader_cfg["pin_memory"]}
    train_loader = DataLoader(
        train_set,
        batch_size=loader_cfg["batch_size"],
        shuffle=not cfg["sampler"].get("class_balanced", False),
        sampler=train_set.build_sampler(cfg["sampler"]),
        drop_last=loader_cfg["drop_last"],
        **common,
    )
    eval_batch_size = loader_cfg.get("eval_batch_size", loader_cfg["batch_size"])
    val_loader = DataLoader(val_set, batch_size=eval_batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=eval_batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, {"runtime.device": args.device, "runtime.output_dir": args.output_dir})
    output_dir = Path(cfg["runtime"]["output_dir"])
    ensure_dir(output_dir)
    logger = setup_logger(output_dir / "train.log")
    seed_everything(cfg["runtime"]["seed"], deterministic=cfg["runtime"].get("deterministic", True))

    if args.build_splits:
        SplitBuilder.from_config(cfg).build_and_save(fold=args.fold)
    split_paths = load_or_build_splits(cfg, fold=args.fold)

    train_loader, val_loader, test_loader = build_dataloaders(cfg, split_paths)
    model = LADLNet(
        num_classes=len(train_loader.dataset.classes),
        class_names=train_loader.dataset.classes,
        model_cfg=cfg["model"],
        loss_cfg=cfg["loss"],
    )
    trainer = Trainer(
        config=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=train_loader.dataset.classes,
        output_dir=output_dir,
        logger=logger,
    )
    trainer.fit()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
