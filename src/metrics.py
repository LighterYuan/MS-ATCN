from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _find_attack_label_index(label_mapping: Optional[Dict[int, str]]) -> Optional[int]:
    if not label_mapping:
        return None

    for k, v in label_mapping.items():
        name = str(v).strip().lower()
        if name in {"attack", "attacks", "malicious", "anomaly", "intrusion", "1"}:
            return int(k)
        if "attack" in name and name not in {"normal", "benign"}:
            return int(k)
    return None


def _binary_auc(y_true, y_score, positive_label: int) -> float:
    y_true_bin = (np.asarray(y_true) == int(positive_label)).astype(int)
    return float(roc_auc_score(y_true_bin, np.asarray(y_score)))


def compute_metrics(
    y_true,
    y_pred,
    y_prob=None,
    label_mapping: Optional[Dict[int, str]] = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fp = cm.sum(axis=0) - cm.diagonal()
    tn = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - cm.diagonal())
    fpr_per_class = fp / (fp + tn + 1e-12)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "fpr_macro": float(np.mean(fpr_per_class)),
    }

    attack_idx = _find_attack_label_index(label_mapping)
    if attack_idx is not None and len(labels) == 2:
        # attack-as-positive FPR:
        # normal samples incorrectly predicted as attack / all normal samples
        y_true_attack = y_true == attack_idx
        y_pred_attack = y_pred == attack_idx
        fp_attack = np.sum((~y_true_attack) & y_pred_attack)
        tn_attack = np.sum((~y_true_attack) & (~y_pred_attack))
        metrics["fpr_attack_positive"] = float(fp_attack / (fp_attack + tn_attack + 1e-12))

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                positive_idx = attack_idx if attack_idx is not None else 1
                metrics["auc"] = _binary_auc(y_true, y_prob[:, positive_idx], positive_idx)
                if attack_idx is not None:
                    metrics["auc_attack_positive"] = _binary_auc(y_true, y_prob[:, attack_idx], attack_idx)
            elif y_prob.ndim == 2 and y_prob.shape[1] > 2:
                metrics["auc_ovr_macro"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
        except ValueError:
            # AUC is undefined if y_true contains only one class.
            pass

    return metrics


def save_reports(y_true, y_pred, label_mapping, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    label_mapping = {int(k): str(v) for k, v in label_mapping.items()}
    labels = sorted(label_mapping.keys())
    target_names = [label_mapping[i] for i in labels]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=6,
        zero_division=0,
    )
    (out / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(out / "confusion_matrix.csv")

    with open(out / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
