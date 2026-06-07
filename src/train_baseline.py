from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from .baselines import build_baseline_model
from .data import prepare_dataloaders, set_global_seed
from .metrics import compute_metrics, save_reports


DEEP_BASELINES = {"mlp", "cnn", "lstm", "gru", "tcn"}
ML_BASELINES = {"xgboost", "lightgbm"}


def pick_device(pref: str, force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    pref = str(pref).lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def call_compute_metrics(y_true, y_pred, y_prob=None, label_mapping=None):
    """
    Compatible with both metric versions:
    1. old: compute_metrics(y_true, y_pred)
    2. new: compute_metrics(y_true, y_pred, y_prob=..., label_mapping=...)
    """
    sig = inspect.signature(compute_metrics)
    kwargs = {}
    if "y_prob" in sig.parameters:
        kwargs["y_prob"] = y_prob
    if "label_mapping" in sig.parameters:
        kwargs["label_mapping"] = label_mapping
    return compute_metrics(y_true, y_pred, **kwargs)


def infer_shape(prepared):
    x0, _ = next(iter(prepared.train_loader))
    in_channels = int(x0.shape[1])
    seq_len = int(x0.shape[2])
    num_classes = int(getattr(prepared, "num_classes", len(prepared.label_mapping)))
    return in_channels, seq_len, num_classes


def run_epoch(model, loader, criterion, optimizer, device, grad_clip_norm=None, label_mapping=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for x, y in tqdm(loader, leave=False):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None and float(grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                optimizer.step()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += float(loss.item()) * int(x.size(0))
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
            y_prob.extend(probs.detach().cpu().tolist())

    metrics = call_compute_metrics(y_true, y_pred, y_prob=y_prob, label_mapping=label_mapping)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics, y_true, y_pred, y_prob


def loader_to_numpy(loader):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.detach().cpu().numpy().reshape(x.size(0), -1))
        ys.append(y.detach().cpu().numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _to_numpy_1d_baseline(x):
    """Convert list/array/series/tensor to a 1D numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.reshape(-1)
    return x


def _get_attack_probability_from_model(model, X):
    """
    Return P(class=1) for sklearn/XGBoost/LightGBM-style models.

    This avoids blindly assuming that predict_proba columns are always [0, 1]
    by checking model.classes_ when available.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError("The model does not support predict_proba().")

    proba = np.asarray(model.predict_proba(X))

    if proba.ndim != 2:
        raise ValueError(f"predict_proba output must be 2D, got shape {proba.shape}")

    classes = getattr(model, "classes_", None)
    if classes is not None:
        classes = list(classes)
        if 1 not in classes:
            raise ValueError(
                f"Class label 1 was not found in model.classes_: {classes}. "
                "Binary labels must be 0=benign and 1=attack."
            )
        attack_index = classes.index(1)
        p_attack = proba[:, attack_index]
    else:
        if proba.shape[1] < 2:
            raise ValueError(
                f"predict_proba output has shape {proba.shape}; "
                "expected at least two probability columns."
            )
        p_attack = proba[:, 1]

    p_attack = np.asarray(p_attack).astype(float)
    if np.any(p_attack < 0) or np.any(p_attack > 1):
        raise ValueError("Predicted probabilities must be in [0, 1].")
    return p_attack


def _compute_binary_metrics_from_probability_baseline(y_true, p_attack, threshold=0.5):
    """
    Compute diagnostic metrics from saved probability outputs.

    These diagnostic files are only used for decision refinement checking.
    They do not replace the original metrics.json/classification_report/confusion_matrix.
    """
    y_true = _to_numpy_1d_baseline(y_true).astype(int)
    p_attack = _to_numpy_1d_baseline(p_attack).astype(float)

    if np.any(p_attack < 0) or np.any(p_attack > 1):
        raise ValueError("Predicted probabilities must be in [0, 1].")

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


def save_baseline_probability_outputs_for_dr(
    y_true,
    p_attack,
    output_dir,
    split_name,
    model_name,
    dataset_name=None,
    task_mode="binary",
    threshold=0.5,
    training_scope=None,
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

    y_true = _to_numpy_1d_baseline(y_true).astype(int)
    p_attack = _to_numpy_1d_baseline(p_attack).astype(float)

    if len(y_true) != len(p_attack):
        raise ValueError(
            f"Length mismatch when saving {split_name} predictions: "
            f"len(y_true)={len(y_true)}, len(p_attack)={len(p_attack)}"
        )

    if np.any(p_attack < 0) or np.any(p_attack > 1):
        raise ValueError("Predicted probabilities must be in [0, 1].")

    metrics, cm, y_pred = _compute_binary_metrics_from_probability_baseline(
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
        "training_scope": training_scope,
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


def train_deep(args, config, prepared, output_dir: Path):
    device = pick_device(config.get("device", "auto"), force_cpu=args.cpu)
    in_channels, seq_len, num_classes = infer_shape(prepared)

    hidden_dim = int(args.hidden_dim if args.hidden_dim is not None else config.get("hidden_channels", 64))
    dropout = float(args.dropout if args.dropout is not None else config.get("dropout", 0.1))
    epochs = int(args.epochs if args.epochs is not None else config.get("epochs", 50))
    lr = float(args.lr if args.lr is not None else config.get("learning_rate", 0.0005))
    weight_decay = float(args.weight_decay if args.weight_decay is not None else config.get("weight_decay", 0.0001))
    grad_clip_norm = config.get("grad_clip_norm", 5.0)

    model = build_baseline_model(
        model_name=args.model,
        in_channels=in_channels,
        seq_len=seq_len,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_path = output_dir / "best_model.pt"
    best_val = -1.0

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

        monitor = float(val_metrics.get("f1_macro", val_metrics.get("accuracy", 0.0)))

        print(
            f"[{args.model.upper()}][Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_acc={val_metrics.get('accuracy', 0.0):.6f} "
            f"val_macro_f1={val_metrics.get('f1_macro', 0.0):.6f}"
        )

        if monitor > best_val:
            best_val = monitor
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "in_channels": in_channels,
                    "seq_len": seq_len,
                    "num_classes": num_classes,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "label_mapping": prepared.label_mapping,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, y_true, y_pred, _ = run_epoch(
        model=model,
        loader=prepared.test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        label_mapping=prepared.label_mapping,
    )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    save_reports(y_true, y_pred, prepared.label_mapping, str(output_dir))

    with open(output_dir / "baseline_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "dataset": args.dataset,
                "task_mode": args.task_mode,
                "input": "[batch, features, window]",
                "in_channels": in_channels,
                "seq_len": seq_len,
                "num_classes": num_classes,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "epochs": epochs,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "selection_metric": "validation_macro_f1",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved baseline outputs to: {output_dir}")


def train_ml(args, prepared, output_dir: Path):
    X_train, y_train = loader_to_numpy(prepared.train_loader)
    X_val, y_val = loader_to_numpy(prepared.val_loader)
    X_test, y_test = loader_to_numpy(prepared.test_loader)

    X_fit = np.concatenate([X_train, X_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)

    if args.model == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is not installed. Run: pip install xgboost") from exc

        model = XGBClassifier(
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            learning_rate=float(args.learning_rate),
            subsample=float(args.subsample),
            colsample_bytree=float(args.colsample_bytree),
            objective="binary:logistic" if len(prepared.label_mapping) == 2 else "multi:softprob",
            eval_metric="logloss",
            tree_method="hist",
            random_state=int(args.seed) if args.seed is not None else 42,
            n_jobs=-1,
        )

    elif args.model == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm") from exc

        model = LGBMClassifier(
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            learning_rate=float(args.learning_rate),
            subsample=float(args.subsample),
            colsample_bytree=float(args.colsample_bytree),
            random_state=int(args.seed) if args.seed is not None else 42,
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unsupported ML baseline: {args.model}")

    model.fit(X_fit, y_fit)

    # ------------------------------------------------------------------
    # Probability export for validation-based decision refinement.
    #
    # IMPORTANT:
    # The original baseline below still uses model trained on train+val,
    # preserving the previous metrics.json behavior.
    #
    # For decision-refinement alpha selection, validation probabilities
    # must not come from a model trained on validation samples. Therefore,
    # this auxiliary model is trained on the original training split only.
    # It uses the same prepared data interface and the same hyperparameters.
    # ------------------------------------------------------------------
    if len(prepared.label_mapping) == 2:
        dr_model = clone(model)
        dr_model.fit(X_train, y_train)

        paper_model_name = "XGBoost" if args.model == "xgboost" else "LightGBM"

        val_p_attack = _get_attack_probability_from_model(dr_model, X_val)
        save_baseline_probability_outputs_for_dr(
            y_true=y_val,
            p_attack=val_p_attack,
            output_dir=output_dir,
            split_name="val",
            model_name=paper_model_name,
            dataset_name=args.dataset,
            task_mode=args.task_mode,
            threshold=0.5,
            training_scope="train_only_for_validation_based_decision_refinement",
        )

        test_p_attack = _get_attack_probability_from_model(dr_model, X_test)
        save_baseline_probability_outputs_for_dr(
            y_true=y_test,
            p_attack=test_p_attack,
            output_dir=output_dir,
            split_name="test",
            model_name=paper_model_name,
            dataset_name=args.dataset,
            task_mode=args.task_mode,
            threshold=0.5,
            training_scope="train_only_for_validation_based_decision_refinement",
        )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    metrics = call_compute_metrics(y_test, y_pred, y_prob=y_prob, label_mapping=prepared.label_mapping)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    save_reports(y_test, y_pred, prepared.label_mapping, str(output_dir))

    with open(output_dir / "baseline_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "dataset": args.dataset,
                "task_mode": args.task_mode,
                "input": "flattened_sliding_window",
                "n_estimators": int(args.n_estimators),
                "max_depth": int(args.max_depth),
                "learning_rate": float(args.learning_rate),
                "subsample": float(args.subsample),
                "colsample_bytree": float(args.colsample_bytree),
                "random_state": int(args.seed),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved baseline outputs to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Independent baseline training entry for binary NIDS experiments.")

    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=["unsw_nb15", "cic_ids2017", "nsl_kdd", "custom_csv"])
    parser.add_argument("--task_mode", default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--data_root", default="./datasets")
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--model", required=True, choices=sorted(DEEP_BASELINES | ML_BASELINES))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.9)

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    with open(output_dir / "run_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    set_global_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

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
    print(f"baseline     : {args.model}")
    print(f"dataset      : {args.dataset}")
    print(f"task_mode    : {args.task_mode}")
    print(f"num_features : {prepared.num_features}")
    print(f"seq_len      : {prepared.seq_len}")
    print(f"num_classes  : {prepared.num_classes}")
    print("=" * 80)

    if args.model in DEEP_BASELINES:
        train_deep(args, config, prepared, output_dir)
    elif args.model in ML_BASELINES:
        train_ml(args, prepared, output_dir)
    else:
        raise ValueError(f"Unsupported baseline model: {args.model}")


if __name__ == "__main__":
    main()
