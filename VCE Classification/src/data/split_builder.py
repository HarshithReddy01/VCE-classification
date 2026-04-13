from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from src.data.dataset import infer_frame_number, infer_video_id, scan_class_folders
from src.utils.io import ensure_dir


@dataclass
class SplitBuilder:
    dataset_root: Path
    metadata_path: Path | None
    split_dir: Path
    strategy: str
    group_column: str
    label_column: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    num_folds: int
    random_state: int

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SplitBuilder":
        return cls(
            dataset_root=Path(cfg["dataset"]["root"]),
            metadata_path=Path(cfg["dataset"]["metadata_path"]) if cfg["dataset"].get("metadata_path") else None,
            split_dir=Path(cfg["splits"]["split_dir"]),
            strategy=cfg["splits"]["strategy"],
            group_column=cfg["splits"]["group_column"],
            label_column=cfg["splits"]["label_column"],
            train_ratio=float(cfg["splits"]["train_ratio"]),
            val_ratio=float(cfg["splits"]["val_ratio"]),
            test_ratio=float(cfg["splits"]["test_ratio"]),
            num_folds=int(cfg["splits"]["num_folds"]),
            random_state=int(cfg["splits"]["random_state"]),
        )

    def _load_metadata(self) -> pd.DataFrame:
        if self.metadata_path and self.metadata_path.exists():
            df = pd.read_csv(self.metadata_path)
        else:
            df = scan_class_folders(self.dataset_root)
        if "video_id" not in df.columns:
            df["video_id"] = df["filename"].astype(str).map(infer_video_id)
        if "frame_number" not in df.columns:
            df["frame_number"] = df["filename"].astype(str).map(infer_frame_number)
        return df

    def build_and_save(self, fold: int | None = None) -> dict[str, Path]:
        df = self._load_metadata().reset_index(drop=True)
        ensure_dir(self.split_dir)

        if self.strategy == "stratified_group_kfold":
            fold = 0 if fold is None else fold
            splitter = StratifiedGroupKFold(n_splits=self.num_folds, shuffle=True, random_state=self.random_state)
            train_val_idx, test_idx = list(splitter.split(df, y=df[self.label_column], groups=df[self.group_column]))[fold]
            train_val_df, test_df = df.iloc[train_val_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=self.test_ratio, random_state=self.random_state)
            train_val_idx, test_idx = next(gss.split(df, groups=df[self.group_column]))
            train_val_df, test_df = df.iloc[train_val_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

        val_fraction = self.val_ratio / max(self.train_ratio + self.val_ratio, 1e-8)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=self.random_state + 1)
        train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df[self.group_column]))
        train_df, val_df = train_val_df.iloc[train_idx].reset_index(drop=True), train_val_df.iloc[val_idx].reset_index(drop=True)

        paths = {name: self.split_dir / f"{name}.csv" for name in ("train", "val", "test")}
        train_df.to_csv(paths["train"], index=False)
        val_df.to_csv(paths["val"], index=False)
        test_df.to_csv(paths["test"], index=False)
        return paths


def load_or_build_splits(cfg: dict[str, Any], fold: int | None = None) -> dict[str, Path]:
    split_dir = Path(cfg["splits"]["split_dir"])
    paths = {name: split_dir / f"{name}.csv" for name in ("train", "val", "test")}
    if all(path.exists() for path in paths.values()):
        return paths
    return SplitBuilder.from_config(cfg).build_and_save(fold=fold)
