from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data import (
    _basic_clean,
    _read_cic_ids2017,
    _read_nsl_kdd,
    _read_unsw_nb15,
)

DatasetName = Literal["cic_ids2017", "nsl_kdd", "unsw_nb15"]
LeakageProtocol = Literal["clean", "attack_cat", "id", "both"]
SplitPolicy = Literal["default", "ordered", "cic_day_holdout"]
RecordOrder = Literal["original", "shuffled"]


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class LazyWindowDataset(Dataset):
    """Window dataset that avoids materializing all overlapping windows in RAM."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_len: int, stride: int = 1):
        features = np.ascontiguousarray(features, dtype=np.float32)
        labels = np.ascontiguousarray(labels, dtype=np.int64)
        if features.ndim != 2:
            raise ValueError(f"features must be 2D, got shape={features.shape}")
        if labels.ndim != 1 or len(labels) != len(features):
            raise ValueError("labels must be 1D and aligned with features")
        if seq_len < 1 or stride < 1:
            raise ValueError("seq_len and stride must be positive")
        if len(features) < seq_len:
            raise ValueError(f"rows={len(features)} is smaller than seq_len={seq_len}")
        self.features = features
        self.labels = labels
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.window_labels = labels[self.seq_len - 1 :: self.stride]
        self.n_windows = int(math.floor((len(features) - seq_len) / stride) + 1)
        self.window_labels = self.window_labels[: self.n_windows]

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, index: int):
        start = int(index) * self.stride
        stop = start + self.seq_len
        x = torch.from_numpy(self.features[start:stop]).transpose(0, 1)
        y = torch.tensor(self.labels[stop - 1], dtype=torch.long)
        return x, y

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        view = np.lib.stride_tricks.sliding_window_view(
            self.features, window_shape=self.seq_len, axis=0
        )
        view = view[:: self.stride][: self.n_windows]  # [N, features, seq_len]
        x = np.ascontiguousarray(view.reshape(len(view), -1), dtype=np.float32)
        y = np.ascontiguousarray(self.window_labels, dtype=np.int64)
        return x, y


@dataclass
class PreparedRevisionData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_dataset: LazyWindowDataset
    val_dataset: LazyWindowDataset
    test_dataset: LazyWindowDataset
    num_features: int
    seq_len: int
    num_classes: int
    class_weights: torch.Tensor
    label_mapping: Dict[int, str]
    data_info: Dict[str, object]

    def numpy_split(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        split = split.lower()
        if split == "train":
            return self.train_dataset.to_numpy()
        if split == "val":
            return self.val_dataset.to_numpy()
        if split == "test":
            return self.test_dataset.to_numpy()
        raise ValueError("split must be train, val, or test")


def _raw_binary_labels(df: pd.DataFrame, dataset: DatasetName) -> np.ndarray:
    if dataset == "unsw_nb15":
        values = df["label"].astype(str).str.strip().str.lower()
        return values.isin({"1", "1.0", "attack", "malicious", "anomaly"}).astype(np.int64).to_numpy()
    if dataset == "cic_ids2017":
        values = df["Label"].astype(str).str.strip().str.upper()
        return (values != "BENIGN").astype(np.int64).to_numpy()
    if dataset == "nsl_kdd":
        values = df["label"].astype(str).str.strip().str.replace(".", "", regex=False).str.lower()
        return (values != "normal").astype(np.int64).to_numpy()
    raise ValueError(dataset)


def _label_column(dataset: DatasetName) -> str:
    return {"unsw_nb15": "label", "cic_ids2017": "Label", "nsl_kdd": "label"}[dataset]


def _chronological_cic_files(data_root: str) -> list[Path]:
    root = Path(data_root) / "CIC-IDS2017"
    files = list(root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CIC-IDS2017 CSV files found under {root}")

    weekday_rank = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
    }

    def key(path: Path):
        name = path.name.lower()
        day = next((rank for token, rank in weekday_rank.items() if token in name), 99)
        part = 0
        if "morning" in name:
            part = 0
        elif "afternoon" in name:
            part = 1
        return day, part, name

    return sorted(files, key=key)


def _read_cic_file(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
    return _basic_clean(df)


def load_raw_splits(
    dataset: DatasetName,
    data_root: str,
    val_ratio: float,
    split_policy: SplitPolicy = "default",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Load fixed raw train/validation/test frames before any fitted preprocessing."""
    dataset = str(dataset).lower()
    split_policy = str(split_policy).lower()

    if dataset == "cic_ids2017" and split_policy == "cic_day_holdout":
        files = _chronological_cic_files(data_root)
        train_files = [p for p in files if any(x in p.name.lower() for x in ("monday", "tuesday", "wednesday"))]
        val_files = [p for p in files if "thursday" in p.name.lower()]
        test_files = [p for p in files if "friday" in p.name.lower()]
        if not train_files or not val_files or not test_files:
            raise ValueError(
                "CIC day holdout requires Monday-Wednesday train files, Thursday validation files, "
                "and Friday test files."
            )
        train_df = pd.concat([_read_cic_file(p) for p in train_files], ignore_index=True)
        val_df = pd.concat([_read_cic_file(p) for p in val_files], ignore_index=True)
        test_df = pd.concat([_read_cic_file(p) for p in test_files], ignore_index=True)
        meta = {
            "split_policy": "cic_day_holdout",
            "train_files": [p.name for p in train_files],
            "val_files": [p.name for p in val_files],
            "test_files": [p.name for p in test_files],
        }
        return train_df, val_df, test_df, meta

    if dataset == "cic_ids2017":
        df = _read_cic_ids2017(data_root)
        n = len(df)
        n_train = int(n * (1.0 - 2.0 * val_ratio))
        n_val = int(n * val_ratio)
        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train : n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val :].copy()
        return train_df, val_df, test_df, {
            "split_policy": "sequential_70_15_15_before_preprocessing_and_windowing",
            "note": "This preserves the original merged-file order used by the supplied project.",
        }

    if dataset == "unsw_nb15":
        official_train, official_test = _read_unsw_nb15(data_root)
    elif dataset == "nsl_kdd":
        official_train, official_test = _read_nsl_kdd(data_root)
    else:
        raise ValueError(f"Unsupported dataset={dataset}")

    n_val = max(1, int(len(official_train) * val_ratio))
    if len(official_train) - n_val < 1:
        raise ValueError("Validation ratio leaves no training rows.")
    # Fixed ordered tail split. Unlike the original implementation, preprocessing
    # is fitted after this split and record order is not destroyed before windowing.
    train_df = official_train.iloc[:-n_val].copy()
    val_df = official_train.iloc[-n_val:].copy()
    test_df = official_test.copy()
    return train_df, val_df, test_df, {
        "split_policy": "official_test_plus_ordered_training_tail_validation",
        "note": (
            "The row order in NSL-KDD/UNSW-NB15 is benchmark record order, not guaranteed packet chronology."
        ),
    }


def feature_frame(
    df: pd.DataFrame,
    dataset: DatasetName,
    leakage_protocol: LeakageProtocol = "clean",
) -> pd.DataFrame:
    label_col = _label_column(dataset)
    drop_cols = [label_col]
    if dataset == "nsl_kdd":
        drop_cols.append("difficulty")
    elif dataset == "unsw_nb15":
        if leakage_protocol == "clean":
            drop_cols.extend(["attack_cat", "id"])
        elif leakage_protocol == "attack_cat":
            drop_cols.append("id")
        elif leakage_protocol == "id":
            drop_cols.append("attack_cat")
        elif leakage_protocol == "both":
            pass
        else:
            raise ValueError(f"Unknown leakage_protocol={leakage_protocol}")
    return df.drop(columns=drop_cols, errors="ignore").copy()


def _encode_train_based(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    train_df = train_df.copy()
    val_df = val_df.reindex(columns=train_df.columns, fill_value=np.nan).copy()
    test_df = test_df.reindex(columns=train_df.columns, fill_value=np.nan).copy()
    category_counts: Dict[str, int] = {}

    object_cols = [
        c for c in train_df.columns
        if train_df[c].dtype == object or str(train_df[c].dtype).startswith("string")
    ]
    for col in object_cols:
        train_values = train_df[col].astype("string").fillna("<MISSING>").astype(str)
        categories = pd.Index(pd.unique(train_values))
        mapping = {value: i for i, value in enumerate(categories)}
        category_counts[col] = len(mapping)
        train_df[col] = train_values.map(mapping).astype(np.int64)
        for frame in (val_df, test_df):
            values = frame[col].astype("string").fillna("<MISSING>").astype(str)
            frame[col] = values.map(mapping).fillna(-1).astype(np.int64)
    return train_df, val_df, test_df, category_counts


def _scaler(kind: str):
    kind = str(kind).lower()
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    if kind in {"none", "null", ""}:
        return None
    raise ValueError(f"Unknown scaling={kind}")


def _preprocess_splits(
    dataset: DatasetName,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    leakage_protocol: LeakageProtocol,
    scaling: str,
    record_order: RecordOrder,
    order_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    y_train = _raw_binary_labels(train_df, dataset)
    y_val = _raw_binary_labels(val_df, dataset)
    y_test = _raw_binary_labels(test_df, dataset)

    x_train = feature_frame(train_df, dataset, leakage_protocol)
    x_val = feature_frame(val_df, dataset, leakage_protocol)
    x_test = feature_frame(test_df, dataset, leakage_protocol)
    x_train, x_val, x_test, category_counts = _encode_train_based(x_train, x_val, x_test)

    for frame in (x_train, x_val, x_test):
        for col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)

    unusable = [col for col in x_train.columns if x_train[col].isna().all()]
    if unusable:
        x_train.drop(columns=unusable, inplace=True)
        x_val.drop(columns=unusable, inplace=True, errors="ignore")
        x_test.drop(columns=unusable, inplace=True, errors="ignore")

    medians = x_train.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_train = x_train.fillna(medians).fillna(0.0)
    x_val = x_val.fillna(medians).fillna(0.0)
    x_test = x_test.fillna(medians).fillna(0.0)

    x_train_np = x_train.astype(np.float32).to_numpy()
    x_val_np = x_val.astype(np.float32).to_numpy()
    x_test_np = x_test.astype(np.float32).to_numpy()

    scaler = _scaler(scaling)
    if scaler is not None:
        x_train_np = scaler.fit_transform(x_train_np).astype(np.float32)
        x_val_np = scaler.transform(x_val_np).astype(np.float32)
        x_test_np = scaler.transform(x_test_np).astype(np.float32)

    if record_order == "shuffled":
        rng = np.random.default_rng(order_seed)

        perm = rng.permutation(len(x_train_np))
        x_train_np = x_train_np[perm].copy()
        y_train = np.asarray(y_train)[perm].copy()

        perm = rng.permutation(len(x_val_np))
        x_val_np = x_val_np[perm].copy()
        y_val = np.asarray(y_val)[perm].copy()

        perm = rng.permutation(len(x_test_np))
        x_test_np = x_test_np[perm].copy()
        y_test = np.asarray(y_test)[perm].copy()
    elif record_order != "original":
        raise ValueError(f"Unknown record_order={record_order}")

    meta = {
        "feature_columns": list(x_train.columns),
        "num_features": int(x_train_np.shape[1]),
        "categorical_training_cardinalities": category_counts,
        "dropped_all_nan_columns": unusable,
        "preprocessing_fit_policy": (
            "categorical mappings, medians, and scaler fitted on the final training split only"
        ),
        "record_order": record_order,
        "order_seed": int(order_seed) if record_order == "shuffled" else None,
    }
    return x_train_np, y_train, x_val_np, y_val, x_test_np, y_test, meta


def _make_loader(
    dataset: LazyWindowDataset,
    batch_size: int,
    shuffle: bool,
    weighted_sampling: bool,
    class_weights: np.ndarray,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    if weighted_sampling:
        weights = class_weights[dataset.window_labels]
        sampler = WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def prepare_revision_data(
    dataset: DatasetName,
    data_root: str,
    output_dir: str,
    batch_size: int,
    val_ratio: float,
    scaling: str,
    num_workers: int,
    seed: int,
    weighted_sampling: bool,
    seq_len: int,
    window_stride: int = 1,
    leakage_protocol: LeakageProtocol = "clean",
    split_policy: SplitPolicy = "default",
    record_order: RecordOrder = "original",
    order_seed: int = 2026,
) -> PreparedRevisionData:
    set_reproducible_seed(seed)
    train_df, val_df, test_df, split_meta = load_raw_splits(
        dataset=dataset,
        data_root=data_root,
        val_ratio=val_ratio,
        split_policy=split_policy,
    )
    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        prep_meta,
    ) = _preprocess_splits(
        dataset=dataset,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        leakage_protocol=leakage_protocol,
        scaling=scaling,
        record_order=record_order,
        order_seed=order_seed,
    )

    train_ds = LazyWindowDataset(x_train, y_train, seq_len=seq_len, stride=window_stride)
    val_ds = LazyWindowDataset(x_val, y_val, seq_len=seq_len, stride=window_stride)
    test_ds = LazyWindowDataset(x_test, y_test, seq_len=seq_len, stride=window_stride)

    counts = np.bincount(train_ds.window_labels, minlength=2).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32)

    train_loader = _make_loader(
        train_ds, batch_size, True, weighted_sampling, weights, num_workers, seed
    )
    val_loader = _make_loader(
        val_ds, batch_size, False, False, weights, num_workers, seed
    )
    test_loader = _make_loader(
        test_ds, batch_size, False, False, weights, num_workers, seed
    )

    data_info: Dict[str, object] = {
        "dataset": dataset,
        "task_mode": "binary",
        "leakage_protocol": leakage_protocol,
        "seq_len": int(seq_len),
        "window_stride": int(window_stride),
        "train_rows": int(len(x_train)),
        "val_rows": int(len(x_val)),
        "test_rows": int(len(x_test)),
        "train_windows": int(len(train_ds)),
        "val_windows": int(len(val_ds)),
        "test_windows": int(len(test_ds)),
        "train_class_counts_windows": {"Benign": int(counts[0]), "Attack": int(counts[1])},
        "window_policy": "constructed independently after split; no cross-split window",
        **split_meta,
        **prep_meta,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_info.json").write_text(
        json.dumps(data_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "label_mapping.json").write_text(
        json.dumps({0: "Benign", 1: "Attack"}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return PreparedRevisionData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        num_features=x_train.shape[1],
        seq_len=seq_len,
        num_classes=2,
        class_weights=class_weights,
        label_mapping={0: "Benign", 1: "Attack"},
        data_info=data_info,
    )
