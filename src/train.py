from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from .data import prepare_dataloaders, set_global_seed
from .losses import FocalLoss
from .metrics import compute_metrics, save_reports
from .model import MSATCN


def pick_device(pref: str) -> torch.device:
    pref = str(pref).lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, optimizer, device, grad_clip_norm=None, label_mapping=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []
    y_prob = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, y in tqdm(loader, leave=False):
            x = x.to(device)  # [B, C, L]
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            total_loss += loss.item() * x.size(0)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
            y_prob.extend(probs.detach().cpu().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    metrics = compute_metrics(y_true, y_pred, y_prob=y_prob, label_mapping=label_mapping)
    metrics["loss"] = avg_loss
    return metrics, y_true, y_pred, y_prob


def _to_numpy_1d(x):
    """Convert tensor/list/array to a 1D numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.reshape(-1)
    return x


def _extract_attack_probability(y_prob):
    """
    Extract attack-class probability for binary IDS.

    Supported shapes:
        [N]      : already attack probability
        [N, 1]   : attack probability
        [N, 2+]  : column 1 is P(attack)
    """
    y_prob = np.asarray(y_prob)

    if y_prob.ndim == 1:
        p_attack = y_prob
    elif y_prob.ndim == 2 and y_prob.shape[1] == 1:
        p_attack = y_prob[:, 0]
    elif y_prob.ndim == 2 and y_prob.shape[1] >= 2:
        p_attack = y_prob[:, 1]
    else:
        raise ValueError(f"Unsupported y_prob shape: {y_prob.shape}")

    p_attack = p_attack.astype(float)
    if np.any(p_attack < 0) or np.any(p_attack > 1):
        raise ValueError(
            "Predicted probabilities must be in [0, 1]. "
            "Please check whether run_epoch returns logits instead of probabilities."
        )
    return p_attack


def _compute_binary_metrics_from_probability(y_true, p_attack, threshold=0.5):
    """
    Compute diagnostic metrics from probability outputs.

    These diagnostic files are only used for decision refinement checking.
    They do not replace the original metrics.json/classification_report/confusion_matrix.
    """
    y_true = _to_numpy_1d(y_true).astype(int)
    p_attack = _extract_attack_probability(p_attack)
    y_pred = (p_attack >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    try:
        auc = float(roc_auc_score(y_true, p_attack))
    except ValueError:
        auc = None

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


def save_probability_outputs_for_dr(
    y_true,
    y_prob,
    output_dir,
    split_name,
    model_name="MS-ATCN",
    dataset_name=None,
    task_mode="binary",
    threshold=0.5,
):
    """
    Save val/test posterior probabilities for validation-based decision refinement.

    Added files:
        val_predictions.csv / test_predictions.csv
        val_probability_metrics.json / test_probability_metrics.json
        val_probability_classification_report.txt / test_probability_classification_report.txt
        val_probability_confusion_matrix.csv / test_probability_confusion_matrix.csv

    The original result files are not overwritten:
        metrics.json
        classification_report.txt
        confusion_matrix.csv
    """
    if split_name not in {"val", "test"}:
        raise ValueError("split_name must be 'val' or 'test'.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = _to_numpy_1d(y_true).astype(int)
    p_attack = _extract_attack_probability(y_prob)

    if len(y_true) != len(p_attack):
        raise ValueError(
            f"Length mismatch when saving {split_name} predictions: "
            f"len(y_true)={len(y_true)}, len(p_attack)={len(p_attack)}"
        )

    metrics, cm, y_pred = _compute_binary_metrics_from_probability(
        y_true=y_true,
        p_attack=p_attack,
        threshold=threshold,
    )

    prediction_path = output_dir / f"{split_name}_predictions.csv"
    pd.DataFrame(
        {
            "y_true": y_true,
            "y_prob": p_attack,
            "y_pred": y_pred,
        }
    ).to_csv(prediction_path, index=False, encoding="utf-8-sig")

    metrics_payload = {
        "dataset": dataset_name,
        "task_mode": task_mode,
        "model": model_name,
        "split": split_name,
        "note": (
            "Diagnostic metrics computed from saved probability outputs. "
            "Original metrics.json is not overwritten."
        ),
        **metrics,
    }

    with open(output_dir / f"{split_name}_probability_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Attack"],
        digits=6,
        zero_division=0,
    )
    with open(
        output_dir / f"{split_name}_probability_classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    cm_df = pd.DataFrame(
        cm,
        index=["true_0_benign", "true_1_attack"],
        columns=["pred_0_benign", "pred_1_attack"],
    )
    cm_df.to_csv(
        output_dir / f"{split_name}_probability_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    print(f"[OK] Saved {model_name} {split_name} probability outputs to: {prediction_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["unsw_nb15", "cic_ids2017", "nsl_kdd", "custom_csv"],
    )
    parser.add_argument(
        "--task_mode",
        type=str,
        default="binary",
        choices=["binary", "multiclass"],
    )
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    run_args = vars(args).copy()
    with open(output_dir / "run_args.json", "w", encoding="utf-8") as f:
        json.dump(run_args, f, indent=2, ensure_ascii=False)

    seed = int(config.get("seed", 42))
    set_global_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = pick_device(config.get("device", "auto"))

    prepared = prepare_dataloaders(
        dataset=args.dataset,
        task_mode=args.task_mode,
        data_root=args.data_root,
        csv_path=args.csv_path,
        label_col=args.label_col,
        batch_size=int(config["batch_size"]),
        val_ratio=float(config["val_ratio"]),
        scaling=config.get("feature_scaling", "standard"),
        num_workers=int(config.get("num_workers", 0)),
        seed=seed,
        weighted_sampling=bool(config.get("weighted_sampling", False)),
        seq_len=int(config.get("seq_len", 16)),
        output_dir=str(output_dir),
    )

    print("=" * 80)
    print(f"dataset      : {args.dataset}")
    print(f"task_mode    : {args.task_mode}")
    print(f"num_features : {prepared.num_features}")
    print(f"seq_len      : {prepared.seq_len}")
    print(f"num_classes  : {prepared.num_classes}")
    print(f"device       : {device}")
    print("=" * 80)

    model = MSATCN(
        input_channels=prepared.num_features,
        seq_len=prepared.seq_len,
        num_classes=prepared.num_classes,
        hidden_channels=int(config.get("hidden_channels", 64)),
        num_blocks=int(config.get("num_blocks", 4)),
        dilations=list(config.get("dilations", [1, 2, 4, 8])),
        kernel_sizes=list(config.get("multi_scale_kernel_sizes", [3, 5, 7])),
        dropout=float(config.get("dropout", 0.2)),
        use_channel_attention=bool(config.get("use_channel_attention", True)),
        use_temporal_attention=bool(config.get("use_temporal_attention", True)),
    ).to(device)

    class_weights = prepared.class_weights.to(device)

    if config.get("use_focal_loss", True):
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=float(config.get("focal_gamma", 2.0)),
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(config.get("label_smoothing", 0.0)),
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_score = -1.0
    best_state = None
    history = []
    patience_counter = 0
    start_time = time.time()

    epochs = int(config["epochs"])
    early_patience = int(config.get("early_stopping_patience", 10))
    grad_clip_norm = float(config.get("grad_clip_norm", 5.0))

    for epoch in range(1, epochs + 1):
        train_metrics, _, _, _ = run_epoch(
            model=model,
            loader=prepared.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=grad_clip_norm,
        )

        val_metrics, _, _, _ = run_epoch(
            model=model,
            loader=prepared.val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            label_mapping=prepared.label_mapping,
        )

        scheduler.step(val_metrics["f1_macro"])

        row = {"epoch": epoch}
        for k, v in train_metrics.items():
            row[f"train_{k}"] = v
        for k, v in val_metrics.items():
            row[f"val_{k}"] = v
        history.append(row)

        score = val_metrics["f1_macro"]

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_f1_macro={train_metrics['f1_macro']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_f1_macro={val_metrics['f1_macro']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if score > best_score:
            best_score = score
            patience_counter = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= early_patience:
                print("Early stopping triggered.")
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    if best_state is not None:
        model.load_state_dict(best_state)

    if prepared.num_classes == 2:
        val_metrics_for_dr, val_y_true_for_dr, val_y_pred_for_dr, val_y_prob_for_dr = run_epoch(
            model=model,
            loader=prepared.val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            label_mapping=prepared.label_mapping,
        )
        save_probability_outputs_for_dr(
            y_true=val_y_true_for_dr,
            y_prob=val_y_prob_for_dr,
            output_dir=output_dir,
            split_name="val",
            model_name="MS-ATCN",
            dataset_name=args.dataset,
            task_mode=args.task_mode,
            threshold=0.5,
        )

    test_metrics, y_true, y_pred, y_prob = run_epoch(
        model=model,
        loader=prepared.test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        label_mapping=prepared.label_mapping,
    )

    if prepared.num_classes == 2:
        save_probability_outputs_for_dr(
            y_true=y_true,
            y_prob=y_prob,
            output_dir=output_dir,
            split_name="test",
            model_name="MS-ATCN",
            dataset_name=args.dataset,
            task_mode=args.task_mode,
            threshold=0.5,
        )

    total_runtime = time.time() - start_time
    test_metrics["total_runtime_seconds"] = total_runtime
    test_metrics["dataset"] = args.dataset
    test_metrics["task_mode"] = args.task_mode

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    save_reports(y_true, y_pred, prepared.label_mapping, str(output_dir))

    print("\nFinal test metrics:")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
