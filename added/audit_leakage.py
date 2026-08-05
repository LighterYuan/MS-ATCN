from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .data_protocols import (
    _raw_binary_labels,
    feature_frame,
    load_raw_splits,
)


def stable_row_hash(frame: pd.DataFrame) -> np.ndarray:
    frame = frame.copy()
    for col in frame.columns:
        if frame[col].dtype == object or str(frame[col].dtype).startswith("string"):
            frame[col] = frame[col].astype("string").fillna("<MISSING>").astype(str)
        else:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy(
        dtype=np.uint64
    )


def duplicate_summary(hashes: np.ndarray) -> Dict[str, int | float]:
    unique_count = int(np.unique(hashes).size)
    duplicate_rows = int(len(hashes) - unique_count)
    return {
        "rows": int(len(hashes)),
        "unique_feature_rows": unique_count,
        "duplicate_rows_beyond_first": duplicate_rows,
        "duplicate_fraction": float(duplicate_rows / max(len(hashes), 1)),
    }


def cross_overlap(a: np.ndarray, b: np.ndarray) -> Dict[str, int | float]:
    ua = np.unique(a)
    ub = np.unique(b)
    common = np.intersect1d(ua, ub, assume_unique=True)
    b_rows_matching_a = int(np.isin(b, common, assume_unique=False).sum())
    a_rows_matching_b = int(np.isin(a, common, assume_unique=False).sum())
    return {
        "common_unique_feature_hashes": int(len(common)),
        "rows_in_first_with_hash_in_second": a_rows_matching_b,
        "rows_in_second_with_hash_in_first": b_rows_matching_a,
        "fraction_first_rows_overlapping": float(a_rows_matching_b / max(len(a), 1)),
        "fraction_second_rows_overlapping": float(b_rows_matching_a / max(len(b), 1)),
    }


def conflicting_duplicate_summary(
    hashes: Iterable[np.ndarray], labels: Iterable[np.ndarray]
) -> Dict[str, int]:
    h = np.concatenate(list(hashes))
    y = np.concatenate(list(labels)).astype(np.int8)
    order = np.argsort(h, kind="mergesort")
    h = h[order]
    y = y[order]
    starts = np.r_[0, np.flatnonzero(h[1:] != h[:-1]) + 1]
    ends = np.r_[starts[1:], len(h)]
    sizes = ends - starts
    minima = np.minimum.reduceat(y, starts)
    maxima = np.maximum.reduceat(y, starts)
    conflict = (sizes > 1) & (minima != maxima)
    return {
        "feature_hashes_with_conflicting_labels": int(conflict.sum()),
        "rows_in_conflicting_duplicate_groups": int(sizes[conflict].sum()),
    }


def adjacent_dependence(labels: np.ndarray) -> Dict[str, float | None]:
    labels = np.asarray(labels, dtype=np.float64)
    if len(labels) < 2:
        return {"adjacent_label_agreement": None, "lag1_label_correlation": None}
    agreement = float(np.mean(labels[1:] == labels[:-1]))
    if np.std(labels[:-1]) == 0 or np.std(labels[1:]) == 0:
        corr = None
    else:
        corr = float(np.corrcoef(labels[:-1], labels[1:])[0, 1])
    return {
        "adjacent_label_agreement": agreement,
        "lag1_label_correlation": corr,
    }


def find_session_columns(frame: pd.DataFrame) -> list[str]:
    aliases = {
        "flow_id",
        "flow id",
        "srcip",
        "src_ip",
        "source ip",
        "dstip",
        "dst_ip",
        "destination ip",
        "sport",
        "source port",
        "dsport",
        "destination port",
        "timestamp",
        "time",
        "proto",
        "protocol",
        "protocol_type",
    }
    found = []
    for col in frame.columns:
        normalized = str(col).strip().lower().replace("-", " ").replace("_", " ")
        if normalized in {x.replace("_", " ") for x in aliases}:
            found.append(str(col))
    return found


def session_overlap(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> Dict[str, object]:
    available = find_session_columns(train)
    # A session audit needs at least a source/destination identity pair or Flow ID.
    lower = {c.lower(): c for c in available}
    flow_id = next((c for c in available if c.lower().replace("_", " ") == "flow id"), None)
    src = next((c for c in available if "src" in c.lower() or "source ip" in c.lower()), None)
    dst = next((c for c in available if "dst" in c.lower() or "destination ip" in c.lower()), None)
    proto = next((c for c in available if "proto" in c.lower()), None)
    sport = next((c for c in available if "sport" in c.lower() or "source port" in c.lower()), None)
    dport = next((c for c in available if "dsport" in c.lower() or "destination port" in c.lower()), None)

    if flow_id is not None:
        cols = [flow_id]
    elif src is not None and dst is not None:
        cols = [src, dst] + [c for c in (sport, dport, proto) if c is not None]
    else:
        return {
            "available_candidate_columns": available,
            "status": "not_computable",
            "reason": (
                "The benchmark files do not retain a sufficient flow/session identifier. "
                "Session-level leakage must be stated as a dataset limitation."
            ),
        }

    hashes = [stable_row_hash(frame[cols]) for frame in (train, val, test)]
    return {
        "available_candidate_columns": available,
        "session_key_columns": cols,
        "status": "computed",
        "train_val": cross_overlap(hashes[0], hashes[1]),
        "train_test": cross_overlap(hashes[0], hashes[2]),
        "val_test": cross_overlap(hashes[1], hashes[2]),
    }


def attack_category_purity(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Dict[str, object]:
    combined = pd.concat([train, val, test], ignore_index=True)
    if "attack_cat" not in combined.columns or "label" not in combined.columns:
        return {"available": False}
    cat = combined["attack_cat"].astype("string").fillna("<MISSING>").astype(str)
    y = _raw_binary_labels(combined, "unsw_nb15")
    table = pd.crosstab(cat, y)
    majority_correct = int(table.max(axis=1).sum())
    return {
        "available": True,
        "number_of_categories": int(table.shape[0]),
        "majority_label_accuracy_using_attack_cat_only": float(
            majority_correct / max(len(combined), 1)
        ),
        "category_label_table": {
            str(idx): {"Benign": int(row.get(0, 0)), "Attack": int(row.get(1, 0))}
            for idx, row in table.iterrows()
        },
    }


def audit_dataset(
    dataset: str,
    data_root: str,
    val_ratio: float,
    output_dir: Path,
    split_policy: str = "default",
) -> Dict[str, object]:
    train, val, test, split_meta = load_raw_splits(
        dataset=dataset,
        data_root=data_root,
        val_ratio=val_ratio,
        split_policy=split_policy,
    )
    features = [
        feature_frame(frame, dataset, leakage_protocol="clean")
        for frame in (train, val, test)
    ]
    labels = [_raw_binary_labels(frame, dataset) for frame in (train, val, test)]
    hashes = [stable_row_hash(frame) for frame in features]

    result: Dict[str, object] = {
        "dataset": dataset,
        "split_metadata": split_meta,
        "clean_feature_count": int(features[0].shape[1]),
        "within_split_duplicates": {
            name: duplicate_summary(h)
            for name, h in zip(("train", "validation", "test"), hashes)
        },
        "cross_split_exact_feature_overlap": {
            "train_validation": cross_overlap(hashes[0], hashes[1]),
            "train_test": cross_overlap(hashes[0], hashes[2]),
            "validation_test": cross_overlap(hashes[1], hashes[2]),
        },
        "conflicting_labels_among_identical_feature_rows": conflicting_duplicate_summary(
            hashes, labels
        ),
        "record_order_dependence": {
            name: adjacent_dependence(y)
            for name, y in zip(("train", "validation", "test"), labels)
        },
        "session_level_audit": session_overlap(train, val, test),
        "sliding_window_dependence": {
            "default_window_length": 16,
            "default_stride": 1,
            "shared_rows_between_adjacent_windows": 15,
            "fraction_of_rows_shared_between_adjacent_windows": 15 / 16,
            "cross_split_windows": 0,
            "interpretation": (
                "Stride-1 windows are strongly dependent within each split, but the revised "
                "pipeline constructs windows after splitting, so this dependence does not cross splits."
            ),
        },
    }
    if dataset == "unsw_nb15":
        result["attack_cat_label_purity"] = attack_category_purity(train, val, test)
        result["explicit_feature_protocols"] = {
            "clean": "drop attack_cat and id",
            "attack_cat": "retain attack_cat, drop id",
            "id": "drop attack_cat, retain id",
            "both": "retain attack_cat and id",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{dataset}_{split_policy}_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit duplicate, split, order, and session leakage.")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--output-dir", default="results/revision_required/audit")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cic_ids2017", "nsl_kdd", "unsw_nb15"],
        choices=["cic_ids2017", "nsl_kdd", "unsw_nb15"],
    )
    parser.add_argument("--include-cic-day-holdout", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    rows = []
    for dataset in args.datasets:
        result = audit_dataset(
            dataset, args.data_root, args.val_ratio, out, split_policy="default"
        )
        for split, stats in result["within_split_duplicates"].items():
            rows.append(
                {
                    "dataset": dataset,
                    "split_policy": "default",
                    "split": split,
                    **stats,
                }
            )
    if args.include_cic_day_holdout and "cic_ids2017" in args.datasets:
        result = audit_dataset(
            "cic_ids2017",
            args.data_root,
            args.val_ratio,
            out,
            split_policy="cic_day_holdout",
        )
        for split, stats in result["within_split_duplicates"].items():
            rows.append(
                {
                    "dataset": "cic_ids2017",
                    "split_policy": "cic_day_holdout",
                    "split": split,
                    **stats,
                }
            )
    pd.DataFrame(rows).to_csv(out / "duplicate_summary.csv", index=False)
    print(f"Saved leakage audit outputs to {out}")


if __name__ == "__main__":
    main()
