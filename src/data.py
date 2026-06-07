from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]


NSL_ATTACK_MAP = {
    "back": "dos",
    "land": "dos",
    "neptune": "dos",
    "pod": "dos",
    "smurf": "dos",
    "teardrop": "dos",
    "mailbomb": "dos",
    "apache2": "dos",
    "processtable": "dos",
    "udpstorm": "dos",
    "ipsweep": "probe",
    "nmap": "probe",
    "portsweep": "probe",
    "satan": "probe",
    "mscan": "probe",
    "saint": "probe",
    "ftp_write": "r2l",
    "guess_passwd": "r2l",
    "imap": "r2l",
    "multihop": "r2l",
    "phf": "r2l",
    "spy": "r2l",
    "warezclient": "r2l",
    "warezmaster": "r2l",
    "sendmail": "r2l",
    "named": "r2l",
    "snmpgetattack": "r2l",
    "snmpguess": "r2l",
    "xlock": "r2l",
    "xsnoop": "r2l",
    "worm": "r2l",
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "perl": "u2r",
    "rootkit": "u2r",
    "httptunnel": "u2r",
    "ps": "u2r",
    "sqlattack": "u2r",
    "xterm": "u2r",
}


@dataclass
class PreparedData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_features: int
    seq_len: int
    num_classes: int
    class_weights: torch.Tensor
    label_mapping: Dict[int, str]


class WindowSequenceDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        """
        windows: [N, seq_len, num_features]
        output x: [num_features, seq_len]
        """
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = self.windows[idx].transpose(0, 1)  # [num_features, seq_len]
        y = self.labels[idx]
        return x, y


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _clean_columns(df)
    df = df.replace([np.inf, -np.inf, "Infinity", "-Infinity", "inf", "-inf"], np.nan)
    df = df.dropna(axis=0).reset_index(drop=True)
    return df


def _normalize_text_label(x: object) -> str:
    return str(x).strip().replace(".", "")


def _encode_object_columns(
    train_df: pd.DataFrame,
    other_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Encode categorical/object feature columns using mappings fitted on the
    training feature frame only.

    Previous versions concatenated train and other_df when building category
    mappings. Although this does not use labels, it still lets validation/test
    feature values influence preprocessing metadata. For leakage-aware
    evaluation, mappings are now fitted only on train_df, and unseen categories
    in other_df are encoded as -1.
    """
    encoded = _encode_object_columns_train_based(train_df, other_df)
    return encoded[0], encoded[1] if len(encoded) > 1 else None


def _encode_object_columns_train_based(
    train_df: pd.DataFrame,
    *other_dfs: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, ...]:
    """
    Train-only categorical encoding for one or more additional splits.

    Parameters
    ----------
    train_df:
        Feature dataframe for the training split. Category-to-index mappings
        are fitted exclusively from this dataframe.
    *other_dfs:
        Validation/test feature dataframes to transform with the training
        mappings. Unknown categories are encoded as -1.

    Returns
    -------
    tuple[pd.DataFrame, ...]
        Encoded train dataframe followed by encoded additional dataframes.
    """
    train_df = train_df.copy()
    transformed_others = [None if df is None else df.copy() for df in other_dfs]

    object_cols = [
        c for c in train_df.columns
        if train_df[c].dtype == object or str(train_df[c].dtype).startswith("string")
    ]

    for c in object_cols:
        train_df[c] = train_df[c].astype(str).fillna("missing")
        categories = pd.Index(pd.unique(train_df[c]))
        mapping = {v: i for i, v in enumerate(categories)}

        train_df[c] = train_df[c].map(mapping).astype(np.int64)

        for i, df in enumerate(transformed_others):
            if df is None:
                continue
            if c in df.columns:
                df[c] = df[c].astype(str).fillna("missing")
                df[c] = df[c].map(mapping).fillna(-1).astype(np.int64)
            else:
                df[c] = -1
            transformed_others[i] = df

    return (train_df, *transformed_others)


def _sanitize_unsw_labels(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_label_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    if target_label_col == "label":
        train_df = train_df.drop(columns=["attack_cat"], errors="ignore")
        test_df = test_df.drop(columns=["attack_cat"], errors="ignore")
    elif target_label_col == "attack_cat":
        train_df = train_df.drop(columns=["label"], errors="ignore")
        test_df = test_df.drop(columns=["label"], errors="ignore")

    return train_df, test_df


def _collapse_cic_label_multiclass(x: str) -> str:
    s = str(x).strip()

    if s.upper() == "BENIGN":
        return "BENIGN"

    s_low = s.lower()
    if "web attack" in s_low or "brute force" in s_low or "xss" in s_low or "sql injection" in s_low:
        return "WebAttack"
    if "ftp-patator" in s_low:
        return "FTP-Patator"
    if "ssh-patator" in s_low:
        return "SSH-Patator"
    if "doS".lower() in s_low and "ddos" not in s_low:
        return s.replace("DoS", "DoS").strip()
    if "ddos" in s_low:
        return "DDoS"
    if "portscan" in s_low:
        return "PortScan"
    if "bot" in s_low:
        return "Bot"
    if "infiltration" in s_low:
        return "Infiltration"
    if "heartbleed" in s_low:
        return "Heartbleed"

    return s


def _remap_labels(
    df: pd.DataFrame,
    dataset_name: str,
    label_col: str,
    task_mode: str,
) -> pd.DataFrame:
    df = df.copy()
    df[label_col] = df[label_col].astype(str).map(_normalize_text_label)

    if dataset_name == "unsw_nb15":
        if task_mode == "binary":
            if label_col != "label":
                raise ValueError("UNSW binary mode must use label_col='label'")
            df[label_col] = df[label_col].map(lambda x: "attack" if str(x) in {"1", "attack"} else "normal")
        elif task_mode == "multiclass":
            if label_col != "attack_cat":
                raise ValueError("UNSW multiclass mode must use label_col='attack_cat'")
            df[label_col] = df[label_col].replace({"": np.nan, "nan": np.nan, "NaN": np.nan})
            df = df.dropna(subset=[label_col]).reset_index(drop=True)
            df[label_col] = df[label_col].astype(str).str.strip()

    elif dataset_name == "cic_ids2017":
        if task_mode == "binary":
            df[label_col] = df[label_col].map(lambda x: "BENIGN" if str(x).upper() == "BENIGN" else "ATTACK")
        elif task_mode == "multiclass":
            df[label_col] = df[label_col].map(_collapse_cic_label_multiclass)

    elif dataset_name == "nsl_kdd":
        if task_mode == "binary":
            df[label_col] = df[label_col].map(lambda x: "normal" if str(x) == "normal" else "attack")
        elif task_mode == "multiclass":
            def nsl_map(x: str) -> str:
                x = str(x)
                if x == "normal":
                    return "normal"
                return NSL_ATTACK_MAP.get(x, "other")
            df[label_col] = df[label_col].map(nsl_map)

    return df


def _filter_feature_columns(
    df: pd.DataFrame,
    label_col: str,
    dataset: str,
    task_mode: str,
) -> pd.DataFrame:
    drop_cols = [label_col]

    if dataset == "unsw_nb15":
        # id is a row identifier and should not be used as a predictive feature.
        drop_cols += ["id"]
        if task_mode == "binary":
            # attack_cat must be removed in binary mode to avoid label leakage.
            drop_cols += ["attack_cat"]
        elif task_mode == "multiclass":
            drop_cols += ["label"]

    if dataset == "nsl_kdd":
        drop_cols += ["difficulty"]

    feature_df = df.drop(columns=drop_cols, errors="ignore")
    return feature_df


def _make_windows(
    features: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(features) < seq_len:
        raise ValueError(f"Number of rows ({len(features)}) is smaller than seq_len ({seq_len})")

    xs = []
    ys = []
    for i in range(0, len(features) - seq_len + 1):
        xs.append(features[i:i + seq_len])
        ys.append(labels[i + seq_len - 1])  # use last record label

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def _fit_scaler(kind: str):
    kind = str(kind).lower()
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    return None


def _build_label_encoder(train_labels: np.ndarray, other_labels: Optional[np.ndarray] = None) -> LabelEncoder:
    le = LabelEncoder()
    if other_labels is not None:
        le.fit(np.concatenate([train_labels, other_labels]))
    else:
        le.fit(train_labels)
    return le



def _encode_labels_for_task(
    train_labels: np.ndarray,
    task_mode: str,
    *other_label_arrays: Optional[np.ndarray],
) -> Tuple[np.ndarray, ...]:
    """
    Encode labels with an explicit binary IDS convention.

    For binary intrusion detection, class 0 is always benign/normal and class 1
    is always attack. This avoids LabelEncoder's alphabetical ordering, which can
    otherwise encode ATTACK as 0 and BENIGN as 1. A stable attack-as-positive
    convention is required for FPR, AUC, saved posterior probabilities, and
    decision-refinement experiments.

    For multiclass experiments, LabelEncoder is retained and fitted on all
    evaluation labels only to create a complete reporting mapping.
    """
    task_mode = str(task_mode).lower()
    train_labels = np.asarray(train_labels).astype(str)
    others = [None if arr is None else np.asarray(arr).astype(str) for arr in other_label_arrays]

    if task_mode == "binary":
        normal_tokens = {"benign", "normal", "0"}
        attack_tokens = {"attack", "attacks", "malicious", "anomaly", "intrusion", "1"}

        def encode_one(arr: np.ndarray) -> np.ndarray:
            out = np.empty(len(arr), dtype=np.int64)
            for i, value in enumerate(arr):
                key = str(value).strip().lower()
                if key in normal_tokens:
                    out[i] = 0
                elif key in attack_tokens:
                    out[i] = 1
                else:
                    raise ValueError(
                        f"Unexpected binary label '{value}'. Expected benign/normal/0 or attack/1 after remapping."
                    )
            return out

        encoded = [encode_one(train_labels)]
        encoded.extend(None if arr is None else encode_one(arr) for arr in others)
        label_mapping = {0: "Benign", 1: "Attack"}
        return (*encoded, label_mapping)

    le = _build_label_encoder(
        train_labels,
        np.concatenate([arr for arr in others if arr is not None]) if any(arr is not None for arr in others) else None,
    )
    encoded = [le.transform(train_labels)]
    encoded.extend(None if arr is None else le.transform(arr) for arr in others)
    label_mapping = {int(i): str(name) for i, name in enumerate(le.classes_)}
    return (*encoded, label_mapping)


def _read_unsw_nb15(data_root: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(data_root) / "UNSW-NB15"
    train_path = root / "UNSW_NB15_training-set.csv"
    test_path = root / "UNSW_NB15_testing-set.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"UNSW-NB15 files not found under {root}. "
            f"Expected: {train_path.name}, {test_path.name}"
        )

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    return _basic_clean(train_df), _basic_clean(test_df)


def _read_cic_ids2017(data_root: str) -> pd.DataFrame:
    root = Path(data_root) / "CIC-IDS2017"
    csv_files = sorted(root.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CIC-IDS2017 csv files found under {root}")

    frames = []
    for p in csv_files:
        try:
            df = pd.read_csv(p, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(p, encoding="latin1", low_memory=False)
        frames.append(df)

    df_all = pd.concat(frames, axis=0, ignore_index=True)
    return _basic_clean(df_all)


def _read_nsl_kdd(data_root: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(data_root) / "NSL-KDD"
    train_path = root / "KDDTrain+.txt"
    test_path = root / "KDDTest+.txt"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"NSL-KDD files not found under {root}. Expected KDDTrain+.txt and KDDTest+.txt"
        )

    train_df = pd.read_csv(train_path, names=NSL_KDD_COLUMNS)
    test_df = pd.read_csv(test_path, names=NSL_KDD_COLUMNS)

    return _basic_clean(train_df), _basic_clean(test_df)


def _read_custom_csv(csv_path: str) -> pd.DataFrame:
    return _basic_clean(pd.read_csv(csv_path, low_memory=False))


def _prepare_from_train_test(
    dataset_name: str,
    task_mode: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    batch_size: int,
    val_ratio: float,
    scaling: str,
    num_workers: int,
    seed: int,
    weighted_sampling: bool,
    seq_len: int,
    output_dir: str,
) -> PreparedData:
    train_df = train_df.copy()
    test_df = test_df.copy()

    if dataset_name == "unsw_nb15":
        train_df, test_df = _sanitize_unsw_labels(train_df, test_df, label_col)

    train_df = _remap_labels(train_df, dataset_name, label_col, task_mode)
    test_df = _remap_labels(test_df, dataset_name, label_col, task_mode)

    if label_col not in train_df.columns or label_col not in test_df.columns:
        raise ValueError(f"Label column '{label_col}' not found after remapping.")

    print(f"\n[{dataset_name}] label_col={label_col} task_mode={task_mode}")
    print("Train label distribution:")
    print(train_df[label_col].value_counts(dropna=False))
    print("Test label distribution:")
    print(test_df[label_col].value_counts(dropna=False))

    X_train_df = _filter_feature_columns(train_df, label_col, dataset_name, task_mode)
    X_test_df = _filter_feature_columns(test_df, label_col, dataset_name, task_mode)

    X_train_df, X_test_df = _encode_object_columns_train_based(X_train_df, X_test_df)

    X_train_df = X_train_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_test_df = X_test_df.reindex(columns=X_train_df.columns, fill_value=0)
    X_test_df = X_test_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Important: do not drop rows here. Row-wise dropna can silently reduce
    # the dataset to zero samples after numeric coercion. Instead, remove only
    # unusable all-NaN columns and impute remaining missing values with medians.
    all_nan_cols = [c for c in X_train_df.columns if X_train_df[c].isna().all()]
    if all_nan_cols:
        print(f"Dropping all-NaN feature columns: {all_nan_cols}")
        X_train_df = X_train_df.drop(columns=all_nan_cols)
        X_test_df = X_test_df.drop(columns=all_nan_cols, errors="ignore")

    if X_train_df.shape[0] == 0 or X_test_df.shape[0] == 0:
        raise ValueError(
            f"No samples remain before splitting: train_rows={X_train_df.shape[0]}, "
            f"test_rows={X_test_df.shape[0]}. Check dataset paths and preprocessing."
        )
    if X_train_df.shape[1] == 0:
        raise ValueError("No feature columns remain after filtering. Check label_col and dataset format.")

    medians = X_train_df.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_train_df = X_train_df.fillna(medians).fillna(0.0)
    X_test_df = X_test_df.fillna(medians).fillna(0.0)

    y_train_raw = train_df.loc[X_train_df.index, label_col].astype(str).values
    y_test_raw = test_df.loc[X_test_df.index, label_col].astype(str).values

    X_train_np = X_train_df.astype(np.float32).values
    X_test_np = X_test_df.astype(np.float32).values

    if len(X_train_np) < max(2, seq_len):
        raise ValueError(
            f"Training data is too small after preprocessing: n_train={len(X_train_np)}, "
            f"seq_len={seq_len}. Check dataset files or reduce seq_len."
        )

    y_train, y_test, label_mapping = _encode_labels_for_task(
        y_train_raw, task_mode, y_test_raw
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_np,
        y_train,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_train,
    )

    scaler = _fit_scaler(scaling)
    if scaler is not None:
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        X_test_np = scaler.transform(X_test_np)

    X_tr_win, y_tr_win = _make_windows(X_tr, y_tr, seq_len)
    X_val_win, y_val_win = _make_windows(X_val, y_val, seq_len)
    X_test_win, y_test_win = _make_windows(X_test_np, y_test, seq_len)

    train_ds = WindowSequenceDataset(X_tr_win, y_tr_win)
    val_ds = WindowSequenceDataset(X_val_win, y_val_win)
    test_ds = WindowSequenceDataset(X_test_win, y_test_win)

    unique, counts = np.unique(y_tr_win, return_counts=True)
    class_count = np.zeros(len(label_mapping), dtype=np.float32)
    class_count[unique] = counts
    class_weights = class_count.sum() / np.maximum(class_count, 1.0)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

    sampler = None
    shuffle = True
    if weighted_sampling:
        sample_weights = class_weights[y_tr_win]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)

    feature_info = {
        "dataset": dataset_name,
        "task_mode": task_mode,
        "label_col": label_col,
        "num_raw_train_rows": int(len(train_df)),
        "num_raw_test_rows": int(len(test_df)),
        "num_features": int(X_tr_win.shape[2]),
        "seq_len": int(seq_len),
        "train_windows": int(len(train_ds)),
        "val_windows": int(len(val_ds)),
        "test_windows": int(len(test_ds)),
        "feature_columns": list(X_train_df.columns),
        "split_policy": "official_train_test_then_train_validation_split",
        "window_policy": "windows_constructed_independently_within_train_val_test_splits",
        "preprocessing_fit_policy": "categorical_mappings_medians_and_scaler_fit_on_training_split_only",
    }
    with open(Path(output_dir) / "data_info.json", "w", encoding="utf-8") as f:
        json.dump(feature_info, f, indent=2, ensure_ascii=False)

    return PreparedData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_features=X_tr_win.shape[2],
        seq_len=seq_len,
        num_classes=len(label_mapping),
        class_weights=class_weights_t,
        label_mapping=label_mapping,
    )


def _prepare_from_single_dataframe(
    dataset_name: str,
    task_mode: str,
    df: pd.DataFrame,
    label_col: str,
    batch_size: int,
    val_ratio: float,
    scaling: str,
    num_workers: int,
    seed: int,
    weighted_sampling: bool,
    seq_len: int,
    output_dir: str,
) -> PreparedData:
    """
    Prepare dataloaders for datasets supplied as a single dataframe.

    Leakage-aware split order
    -------------------------
    The previous implementation constructed sliding windows on the full
    dataframe and then split the window array. That can create highly
    overlapping windows around train/validation/test boundaries. This version
    first performs a deterministic row-level split, then fits preprocessing
    objects only on the training split, and finally constructs windows
    independently inside each split. Therefore, no window can cross a
    train/validation/test boundary.

    The public interface and returned DataLoader shapes remain unchanged:
    every dataset item is still returned as x=[num_features, seq_len], y=label.
    Consequently, MS-ATCN and the baseline scripts remain compatible. Tree
    baselines still receive flattened window-level samples via loader_to_numpy().
    """
    df = df.copy()
    df = _remap_labels(df, dataset_name, label_col, task_mode)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found after remapping.")

    print(f"\n[{dataset_name}] label_col={label_col} task_mode={task_mode}")
    print("Label distribution:")
    print(df[label_col].value_counts(dropna=False))

    X_df = _filter_feature_columns(df, label_col, dataset_name, task_mode)
    y_raw_all = df.loc[X_df.index, label_col].astype(str).values

    n_rows = len(X_df)
    n_train_rows = int(n_rows * (1.0 - 2 * val_ratio))
    n_val_rows = int(n_rows * val_ratio)
    n_test_rows = n_rows - n_train_rows - n_val_rows

    if min(n_train_rows, n_val_rows, n_test_rows) < seq_len:
        raise ValueError(
            "Each split must contain at least seq_len rows before windowing. "
            f"Got train_rows={n_train_rows}, val_rows={n_val_rows}, "
            f"test_rows={n_test_rows}, seq_len={seq_len}. "
            "Reduce seq_len/val_ratio or provide more data."
        )

    # Deterministic row-level split before any scaler fitting or windowing.
    # For CIC-IDS2017 this preserves the original merged-file row order and
    # prevents overlap across split boundaries.
    X_train_df = X_df.iloc[:n_train_rows].copy()
    X_val_df = X_df.iloc[n_train_rows:n_train_rows + n_val_rows].copy()
    X_test_df = X_df.iloc[n_train_rows + n_val_rows:].copy()

    y_train_raw = y_raw_all[:n_train_rows]
    y_val_raw = y_raw_all[n_train_rows:n_train_rows + n_val_rows]
    y_test_raw = y_raw_all[n_train_rows + n_val_rows:]

    X_train_df, X_val_df, X_test_df = _encode_object_columns_train_based(
        X_train_df, X_val_df, X_test_df
    )

    X_train_df = X_train_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_val_df = X_val_df.reindex(columns=X_train_df.columns, fill_value=0)
    X_val_df = X_val_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_test_df = X_test_df.reindex(columns=X_train_df.columns, fill_value=0)
    X_test_df = X_test_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Drop only columns that are unusable in the training split. This preserves
    # row order and sample count while avoiding train/val/test information flow.
    all_nan_cols = [c for c in X_train_df.columns if X_train_df[c].isna().all()]
    if all_nan_cols:
        print(f"Dropping all-NaN feature columns based on training split: {all_nan_cols}")
        X_train_df = X_train_df.drop(columns=all_nan_cols)
        X_val_df = X_val_df.drop(columns=all_nan_cols, errors="ignore")
        X_test_df = X_test_df.drop(columns=all_nan_cols, errors="ignore")

    if X_train_df.shape[1] == 0:
        raise ValueError("No feature columns remain after filtering. Check label_col and dataset format.")

    # Median imputation is fitted on training split only.
    medians = X_train_df.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_train_df = X_train_df.fillna(medians).fillna(0.0)
    X_val_df = X_val_df.fillna(medians).fillna(0.0)
    X_test_df = X_test_df.fillna(medians).fillna(0.0)

    X_train_np = X_train_df.astype(np.float32).values
    X_val_np = X_val_df.astype(np.float32).values
    X_test_np = X_test_df.astype(np.float32).values

    # Use an explicit binary convention: 0=Benign/Normal and 1=Attack.
    # This is essential for attack-as-positive FPR, AUC, posterior saving,
    # and validation-based decision refinement.
    y_train, y_val, y_test, label_mapping = _encode_labels_for_task(
        y_train_raw, task_mode, y_val_raw, y_test_raw
    )

    scaler = _fit_scaler(scaling)
    if scaler is not None:
        X_train_np = scaler.fit_transform(X_train_np)
        X_val_np = scaler.transform(X_val_np)
        X_test_np = scaler.transform(X_test_np)

    # Construct sliding windows independently inside each split.
    # No window can include records from another split.
    X_tr, y_tr = _make_windows(X_train_np, y_train, seq_len)
    X_val, y_val = _make_windows(X_val_np, y_val, seq_len)
    X_test, y_test = _make_windows(X_test_np, y_test, seq_len)

    train_ds = WindowSequenceDataset(X_tr, y_tr)
    val_ds = WindowSequenceDataset(X_val, y_val)
    test_ds = WindowSequenceDataset(X_test, y_test)

    unique, counts = np.unique(y_tr, return_counts=True)
    class_count = np.zeros(len(label_mapping), dtype=np.float32)
    class_count[unique] = counts
    class_weights = class_count.sum() / np.maximum(class_count, 1.0)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

    sampler = None
    shuffle = True
    if weighted_sampling:
        sample_weights = class_weights[y_tr]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)

    feature_info = {
        "dataset": dataset_name,
        "task_mode": task_mode,
        "label_col": label_col,
        "num_raw_rows": int(len(df)),
        "num_train_rows_before_windowing": int(len(X_train_np)),
        "num_val_rows_before_windowing": int(len(X_val_np)),
        "num_test_rows_before_windowing": int(len(X_test_np)),
        "num_features": int(X_tr.shape[2]),
        "seq_len": int(seq_len),
        "train_windows": int(len(train_ds)),
        "val_windows": int(len(val_ds)),
        "test_windows": int(len(test_ds)),
        "feature_columns": list(X_train_df.columns),
        "split_policy": "row_level_sequential_split_before_preprocessing_and_windowing",
        "window_policy": "windows_constructed_independently_within_train_val_test_splits",
        "preprocessing_fit_policy": "categorical_mappings_medians_and_scaler_fit_on_training_split_only",
    }
    with open(Path(output_dir) / "data_info.json", "w", encoding="utf-8") as f:
        json.dump(feature_info, f, indent=2, ensure_ascii=False)

    return PreparedData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_features=X_tr.shape[2],
        seq_len=seq_len,
        num_classes=len(label_mapping),
        class_weights=class_weights_t,
        label_mapping=label_mapping,
    )


def prepare_dataloaders(
    dataset: str,
    task_mode: str,
    data_root: str,
    csv_path: Optional[str],
    label_col: Optional[str],
    batch_size: int,
    val_ratio: float,
    scaling: str,
    num_workers: int,
    seed: int,
    weighted_sampling: bool,
    seq_len: int,
    output_dir: str,
) -> PreparedData:
    set_global_seed(seed)

    dataset = str(dataset).lower()
    task_mode = str(task_mode).lower()

    if dataset == "unsw_nb15":
        train_df, test_df = _read_unsw_nb15(data_root)
        resolved_label = label_col or ("label" if task_mode == "binary" else "attack_cat")
        return _prepare_from_train_test(
            dataset_name=dataset,
            task_mode=task_mode,
            train_df=train_df,
            test_df=test_df,
            label_col=resolved_label,
            batch_size=batch_size,
            val_ratio=val_ratio,
            scaling=scaling,
            num_workers=num_workers,
            seed=seed,
            weighted_sampling=weighted_sampling,
            seq_len=seq_len,
            output_dir=output_dir,
        )

    if dataset == "cic_ids2017":
        df = _read_cic_ids2017(data_root)
        resolved_label = label_col or "Label"
        return _prepare_from_single_dataframe(
            dataset_name=dataset,
            task_mode=task_mode,
            df=df,
            label_col=resolved_label,
            batch_size=batch_size,
            val_ratio=val_ratio,
            scaling=scaling,
            num_workers=num_workers,
            seed=seed,
            weighted_sampling=weighted_sampling,
            seq_len=seq_len,
            output_dir=output_dir,
        )

    if dataset == "nsl_kdd":
        train_df, test_df = _read_nsl_kdd(data_root)
        resolved_label = label_col or "label"
        return _prepare_from_train_test(
            dataset_name=dataset,
            task_mode=task_mode,
            train_df=train_df,
            test_df=test_df,
            label_col=resolved_label,
            batch_size=batch_size,
            val_ratio=val_ratio,
            scaling=scaling,
            num_workers=num_workers,
            seed=seed,
            weighted_sampling=weighted_sampling,
            seq_len=seq_len,
            output_dir=output_dir,
        )

    if dataset == "custom_csv":
        if not csv_path or not label_col:
            raise ValueError("custom_csv mode requires --csv_path and --label_col")
        df = _read_custom_csv(csv_path)
        return _prepare_from_single_dataframe(
            dataset_name=dataset,
            task_mode=task_mode,
            df=df,
            label_col=label_col,
            batch_size=batch_size,
            val_ratio=val_ratio,
            scaling=scaling,
            num_workers=num_workers,
            seed=seed,
            weighted_sampling=weighted_sampling,
            seq_len=seq_len,
            output_dir=output_dir,
        )

    raise ValueError(f"Unsupported dataset: {dataset}")
