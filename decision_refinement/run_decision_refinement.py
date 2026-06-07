#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validation-based Decision Refinement for MS-ATCN.

This script is designed for the paper framework, not for post-hoc score chasing.

Core principle
--------------
1. MS-ATCN remains the main proposed neural model.
2. XGBoost / LightGBM are used only as tree-based decision refinement modules.
3. The refinement weight alpha is selected ONLY on the validation set.
4. The test set is evaluated ONCE using the selected validation alpha.
5. All models must use the same preprocessing, same feature columns, and same split protocol.

Fusion formula
--------------
    p_refined = alpha * p_ms_atcn + (1 - alpha) * p_tree

where:
    p_ms_atcn: attack probability predicted by MS-ATCN
    p_tree: attack probability predicted by XGBoost or LightGBM
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


def read_predictions(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path)
    required = {"y_true", "y_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} must contain columns {sorted(required)}. "
            f"Missing columns: {sorted(missing)}"
        )

    out = df[["y_true", "y_prob"]].copy()
    out["y_true"] = out["y_true"].astype(int)
    out["y_prob"] = out["y_prob"].astype(float)

    if out["y_true"].nunique() < 2:
        raise ValueError(f"{path} contains only one class in y_true.")

    if out["y_prob"].min() < 0 or out["y_prob"].max() > 1:
        raise ValueError(f"{path}: y_prob must be in [0, 1].")

    return out


def check_alignment(ms_df: pd.DataFrame, tree_df: pd.DataFrame, split_name: str):
    if len(ms_df) != len(tree_df):
        raise ValueError(
            f"{split_name} length mismatch: "
            f"MS-ATCN={len(ms_df)}, tree={len(tree_df)}"
        )

    if not np.array_equal(ms_df["y_true"].values, tree_df["y_true"].values):
        raise ValueError(
            f"{split_name} y_true mismatch. "
            "MS-ATCN and tree predictions must be generated on the same split "
            "with the same row order and same preprocessing pipeline."
        )


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    auc = None
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        pass

    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1-score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "FPR": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "attack-as-positive FPR": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "AUC": auc,
        "threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    return metrics, cm, y_pred


def is_higher_better(metric_name: str) -> bool:
    return metric_name not in {"FPR", "attack-as-positive FPR"}


def select_alpha_on_validation(y_true, p_ms, p_tree, alphas, select_metric, threshold):
    rows = []
    best_row = None

    for alpha in alphas:
        p_refined = alpha * p_ms + (1.0 - alpha) * p_tree
        metrics, _, _ = compute_metrics(y_true, p_refined, threshold=threshold)

        row = {
            "alpha_ms_atcn": float(alpha),
            "alpha_tree": float(1.0 - alpha),
            **metrics,
        }
        rows.append(row)

        value = metrics[select_metric]
        if value is None:
            continue

        if best_row is None:
            best_row = row
        elif is_higher_better(select_metric):
            if value > best_row[select_metric]:
                best_row = row
        else:
            if value < best_row[select_metric]:
                best_row = row

    if best_row is None:
        raise RuntimeError(f"Could not select alpha using metric: {select_metric}")

    return pd.DataFrame(rows), best_row


def save_final_outputs(
    out_dir,
    dataset,
    model_name,
    selected_alpha,
    select_metric,
    y_true,
    y_prob,
    y_pred,
    metrics,
    cm,
    args,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "dataset": dataset,
        "model": model_name,
        "decision_refinement": True,
        "alpha_selection_split": "validation",
        "selection_metric": select_metric,
        "selected_alpha_ms_atcn": float(selected_alpha),
        "selected_alpha_tree": float(1.0 - selected_alpha),
        **metrics,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4, ensure_ascii=False)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Attack"],
        digits=6,
        zero_division=0,
    )
    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm_df = pd.DataFrame(
        cm,
        index=["true_0_benign", "true_1_attack"],
        columns=["pred_0_benign", "pred_1_attack"],
    )
    cm_df.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    pd.DataFrame(
        {"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred}
    ).to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Validation-based decision refinement for the proposed MS-ATCN framework.",
        "protocol_constraints": [
            "Same preprocessing pipeline and split protocol for MS-ATCN and tree models.",
            "Alpha is selected only on the validation set.",
            "Test set is evaluated once with the selected alpha.",
            "No test-set alpha search.",
            "UNSW-NB15 must remove attack_cat and id before model training and prediction generation.",
        ],
        "command_arguments": vars(args),
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tree-name", required=True, choices=["LightGBM", "XGBoost"])
    parser.add_argument("--ms-val", required=True)
    parser.add_argument("--tree-val", required=True)
    parser.add_argument("--ms-test", required=True)
    parser.add_argument("--tree-test", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--alphas",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated alpha values. alpha is the MS-ATCN weight.",
    )
    parser.add_argument(
        "--select-metric",
        default="Macro-F1",
        choices=[
            "Accuracy", "Precision", "Recall", "F1-score", "Macro-F1",
            "FPR", "attack-as-positive FPR", "AUC",
        ],
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ms_val = read_predictions(args.ms_val)
    tree_val = read_predictions(args.tree_val)
    ms_test = read_predictions(args.ms_test)
    tree_test = read_predictions(args.tree_test)

    check_alignment(ms_val, tree_val, "validation")
    check_alignment(ms_test, tree_test, "test")

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    val_summary, best_row = select_alpha_on_validation(
        y_true=ms_val["y_true"].values,
        p_ms=ms_val["y_prob"].values,
        p_tree=tree_val["y_prob"].values,
        alphas=alphas,
        select_metric=args.select_metric,
        threshold=args.threshold,
    )

    val_summary.to_csv(out_dir / "val_alpha_search.csv", index=False, encoding="utf-8-sig")

    selected_alpha = float(best_row["alpha_ms_atcn"])
    selected = {
        "dataset": args.dataset,
        "tree_name": args.tree_name,
        "selection_split": "validation",
        "selection_metric": args.select_metric,
        "selected_alpha_ms_atcn": selected_alpha,
        "selected_alpha_tree": float(1.0 - selected_alpha),
        "validation_metrics_at_selected_alpha": best_row,
    }
    with open(out_dir / "selected_alpha.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=4, ensure_ascii=False)

    p_test = selected_alpha * ms_test["y_prob"].values + (1.0 - selected_alpha) * tree_test["y_prob"].values
    y_test = ms_test["y_true"].values
    test_metrics, test_cm, test_pred = compute_metrics(y_test, p_test, threshold=args.threshold)

    model_name = f"MS-ATCN-{args.tree_name}-DR"

    save_final_outputs(
        out_dir=out_dir,
        dataset=args.dataset,
        model_name=model_name,
        selected_alpha=selected_alpha,
        select_metric=args.select_metric,
        y_true=y_test,
        y_prob=p_test,
        y_pred=test_pred,
        metrics=test_metrics,
        cm=test_cm,
        args=args,
    )

    print(f"[OK] Model: {model_name}")
    print(f"[OK] Alpha selected on validation: {selected_alpha:.4f}")
    print(f"[OK] Selection metric: {args.select_metric}")
    print(f"[OK] Test metrics saved to: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
