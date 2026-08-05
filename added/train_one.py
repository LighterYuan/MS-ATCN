from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.baselines import build_baseline_model
from src.metrics import compute_metrics
from src.model import MSATCN

from .data_protocols import PreparedRevisionData, prepare_revision_data, set_reproducible_seed
from .losses import FormulaConsistentFocalLoss

DEEP_MODELS = {"ms_atcn", "mlp", "cnn", "lstm", "gru", "tcn"}
TREE_MODELS = {"xgboost", "lightgbm"}
VARIANTS = {"full", "no_multiscale", "no_attention", "no_focal_loss"}


def pick_device(preference: str, force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    preference = str(preference).lower()
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_reports(y_true, y_pred, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Attack"],
        digits=6,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["true_Benign", "true_Attack"],
        columns=["pred_Benign", "pred_Attack"],
    ).to_csv(output_dir / "confusion_matrix.csv")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float | None = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            total_loss += float(loss.item()) * int(x.size(0))
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
            y_prob.extend(probs.detach().cpu().tolist())

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    y_prob_np = np.asarray(y_prob, dtype=np.float64)
    metrics = compute_metrics(
        y_true_np,
        y_pred_np,
        y_prob=y_prob_np,
        label_mapping={0: "Benign", 1: "Attack"},
    )
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics, y_true_np, y_pred_np, y_prob_np


def build_deep_model(
    model_name: str,
    prepared: PreparedRevisionData,
    config: dict,
    variant: str,
) -> nn.Module:
    if model_name == "ms_atcn":
        kernels = list(config.get("multi_scale_kernel_sizes", [3, 5, 7]))
        use_channel = bool(config.get("use_channel_attention", True))
        use_temporal = bool(config.get("use_temporal_attention", True))
        if variant == "no_multiscale":
            kernels = [3]
        elif variant == "no_attention":
            use_channel = False
            use_temporal = False
        elif variant not in {"full", "no_focal_loss"}:
            raise ValueError(f"Unsupported MS-ATCN variant={variant}")
        return MSATCN(
            input_channels=prepared.num_features,
            seq_len=prepared.seq_len,
            num_classes=2,
            hidden_channels=int(config.get("hidden_channels", 64)),
            num_blocks=int(config.get("num_blocks", 4)),
            dilations=list(config.get("dilations", [1, 2, 4, 8])),
            kernel_sizes=kernels,
            dropout=float(config.get("dropout", 0.1)),
            use_channel_attention=use_channel,
            use_temporal_attention=use_temporal,
        )
    if variant != "full":
        raise ValueError("Ablation variants are only defined for ms_atcn.")
    return build_baseline_model(
        model_name=model_name,
        in_channels=prepared.num_features,
        seq_len=prepared.seq_len,
        num_classes=2,
        hidden_dim=int(config.get("baseline_hidden_dim", config.get("hidden_channels", 64))),
        dropout=float(config.get("baseline_dropout", config.get("dropout", 0.1))),
    )


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def estimate_conv_linear_macs(model: nn.Module, sample: torch.Tensor) -> int:
    """Counts Conv1d and Linear MACs for one forward pass; recurrent cell MACs are not included."""
    macs = 0
    hooks = []

    def conv_hook(module: nn.Conv1d, inputs, output):
        nonlocal macs
        out = output
        batch = int(out.shape[0])
        out_channels = int(out.shape[1])
        out_length = int(out.shape[2])
        per_output = (module.in_channels // module.groups) * module.kernel_size[0]
        macs += batch * out_channels * out_length * per_output

    def linear_hook(module: nn.Linear, inputs, output):
        nonlocal macs
        out = output
        macs += int(out.numel()) * int(module.in_features)

    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        model(sample)
    if was_training:
        model.train()
    for hook in hooks:
        hook.remove()
    return int(macs)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_torch_model(
    model: nn.Module,
    prepared: PreparedRevisionData,
    device: torch.device,
    repetitions: int = 100,
    warmup: int = 20,
) -> Dict[str, object]:
    model.eval()
    first_batch, _ = next(iter(prepared.test_loader))
    batch_full = first_batch.to(device)
    batch_one = batch_full[:1]
    results: Dict[str, object] = {}

    for name, batch in (("batch_1", batch_one), ("batch_default", batch_full)):
        with torch.inference_mode():
            for _ in range(warmup):
                model(batch)
            _synchronize(device)
            start = time.perf_counter()
            for _ in range(repetitions):
                model(batch)
            _synchronize(device)
            elapsed = time.perf_counter() - start
        batch_seconds = elapsed / repetitions
        results[name] = {
            "batch_size": int(batch.shape[0]),
            "mean_batch_latency_ms": float(batch_seconds * 1000.0),
            "mean_latency_ms_per_sample": float(batch_seconds * 1000.0 / batch.shape[0]),
            "throughput_samples_per_second": float(batch.shape[0] / batch_seconds),
            "warmup_iterations": warmup,
            "measurement_iterations": repetitions,
        }

    macs = estimate_conv_linear_macs(model, batch_one)
    results["conv_linear_macs_per_sample"] = macs
    results["approx_conv_linear_flops_per_sample"] = int(2 * macs)
    results["macs_scope_note"] = (
        "Conv1d and Linear layers only. Recurrent-cell operations are not included for LSTM/GRU."
    )
    return results


def environment_info(device: torch.device) -> Dict[str, object]:
    info: Dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cpu_count": os.cpu_count(),
    }
    if device.type == "cuda":
        info.update(
            {
                "cuda": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            }
        )
    return info


def train_deep(
    args: argparse.Namespace,
    config: dict,
    prepared: PreparedRevisionData,
    output_dir: Path,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    model = build_deep_model(args.model, prepared, config, args.variant).to(device)
    class_weights = prepared.class_weights.to(device)

    if args.model == "ms_atcn":
        if args.variant == "no_focal_loss":
            criterion: nn.Module = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=float(config.get("label_smoothing", 0.0)),
            )
        else:
            criterion = FormulaConsistentFocalLoss(
                alpha=class_weights,
                gamma=float(config.get("focal_gamma", 2.0)),
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.get("learning_rate", 5e-4)),
            weight_decay=float(config.get("weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=int(config.get("scheduler_patience", 3))
        )
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config.get("learning_rate", 5e-4)),
            weight_decay=float(config.get("weight_decay", 1e-4)),
        )
        scheduler = None

    epochs = int(config.get("epochs", 50))
    patience = int(config.get("early_stopping_patience", 10))
    grad_clip = float(config.get("grad_clip_norm", 5.0))
    best_score = -np.inf
    best_state = None
    patience_count = 0
    history: list[dict] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_metrics, _, _, _ = _run_epoch(
            model, prepared.train_loader, criterion, device, optimizer, grad_clip
        )
        val_metrics, _, _, _ = _run_epoch(
            model, prepared.val_loader, criterion, device
        )
        score = float(val_metrics["f1_macro"])
        if scheduler is not None:
            scheduler.step(score)

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"val_macro_f1={score:.6f} val_accuracy={val_metrics['accuracy']:.6f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    training_seconds = time.perf_counter() - train_start
    if best_state is None:
        raise RuntimeError("No model checkpoint was selected.")
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    test_start = time.perf_counter()
    test_metrics, y_true, y_pred, y_prob = _run_epoch(
        model, prepared.test_loader, criterion, device
    )
    test_seconds = time.perf_counter() - test_start
    save_reports(y_true, y_pred, output_dir)

    checkpoint_path = output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "model": args.model,
            "variant": args.variant,
            "num_features": prepared.num_features,
            "seq_len": prepared.seq_len,
            "config": config,
        },
        checkpoint_path,
    )
    checkpoint_bytes = checkpoint_path.stat().st_size
    if args.no_save_model:
        checkpoint_path.unlink(missing_ok=True)

    efficiency: Dict[str, object] = {
        "model": args.model,
        "variant": args.variant,
        "trainable_parameters": count_parameters(model),
        "training_seconds": float(training_seconds),
        "test_evaluation_seconds": float(test_seconds),
        "checkpoint_bytes": int(checkpoint_bytes),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "inference": benchmark_torch_model(
            model,
            prepared,
            device,
            repetitions=int(args.latency_repetitions),
            warmup=int(args.latency_warmup),
        ),
    }
    if args.save_predictions:
        np.savez_compressed(
            output_dir / "test_predictions.npz",
            y_true=y_true,
            y_pred=y_pred,
            p_attack=y_prob[:, 1],
        )

    test_metrics.update(
        {
            "seed": int(args.seed),
            "model": args.model,
            "variant": args.variant,
            "dataset": args.dataset,
            "seq_len": int(args.seq_len),
            "leakage_protocol": args.leakage_protocol,
            "split_policy": args.split_policy,
            "record_order": args.record_order,
            "training_seconds": float(training_seconds),
            "hidden_channels": int(config.get("hidden_channels", 64)),
            "focal_gamma": float(config.get("focal_gamma", 2.0)),
            "dropout": float(config.get("dropout", 0.1)),
            "learning_rate": float(config.get("learning_rate", 5e-4)),
        }
    )
    return test_metrics, efficiency


def build_tree_model(model_name: str, args: argparse.Namespace):
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("Install xgboost before running tree baselines.") from exc
        return XGBClassifier(
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            learning_rate=float(args.tree_learning_rate),
            subsample=float(args.subsample),
            colsample_bytree=float(args.colsample_bytree),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=int(args.seed),
            n_jobs=int(args.tree_jobs),
        )
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("Install lightgbm before running tree baselines.") from exc
        return LGBMClassifier(
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            learning_rate=float(args.tree_learning_rate),
            subsample=float(args.subsample),
            subsample_freq=1,
            colsample_bytree=float(args.colsample_bytree),
            random_state=int(args.seed),
            n_jobs=int(args.tree_jobs),
            verbosity=-1,
        )
    raise ValueError(model_name)


def train_tree(
    args: argparse.Namespace,
    prepared: PreparedRevisionData,
    output_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    # Extract from datasets, not the weighted-sampling DataLoader. This prevents
    # duplicate/omitted training windows in the tree baseline.
    x_train, y_train = prepared.numpy_split("train")
    x_test, y_test = prepared.numpy_split("test")
    model = build_tree_model(args.model, args)

    start = time.perf_counter()
    model.fit(x_train, y_train)
    training_seconds = time.perf_counter() - start

    start = time.perf_counter()
    y_prob = np.asarray(model.predict_proba(x_test))
    prediction_seconds = time.perf_counter() - start
    y_pred = np.asarray(model.classes_)[np.argmax(y_prob, axis=1)].astype(np.int64)
    metrics = compute_metrics(
        y_test,
        y_pred,
        y_prob=y_prob,
        label_mapping={0: "Benign", 1: "Attack"},
    )
    save_reports(y_test, y_pred, output_dir)

    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path, compress=3)
    model_bytes = model_path.stat().st_size
    if args.no_save_model:
        model_path.unlink(missing_ok=True)

    n_trees = None
    n_leaves = None
    if args.model == "xgboost":
        try:
            dump = model.get_booster().get_dump()
            n_trees = len(dump)
        except Exception:
            pass
    else:
        try:
            n_trees = int(model.booster_.num_trees())
            tree_info = model.booster_.dump_model().get("tree_info", [])
            n_leaves = int(sum(t.get("num_leaves", 0) for t in tree_info))
        except Exception:
            pass

    efficiency = {
        "model": args.model,
        "variant": "full",
        "training_seconds": float(training_seconds),
        "prediction_seconds": float(prediction_seconds),
        "test_samples": int(len(x_test)),
        "mean_latency_ms_per_sample": float(prediction_seconds * 1000.0 / len(x_test)),
        "throughput_samples_per_second": float(len(x_test) / prediction_seconds),
        "serialized_model_bytes": int(model_bytes),
        "number_of_trees": n_trees,
        "total_leaves": n_leaves,
        "input_dimension": int(x_train.shape[1]),
        "training_rows": int(len(x_train)),
    }
    if args.save_predictions:
        np.savez_compressed(
            output_dir / "test_predictions.npz",
            y_true=y_test,
            y_pred=y_pred,
            p_attack=y_prob[:, list(model.classes_).index(1)],
        )
    metrics.update(
        {
            "seed": int(args.seed),
            "model": args.model,
            "variant": "full",
            "dataset": args.dataset,
            "seq_len": int(args.seq_len),
            "leakage_protocol": args.leakage_protocol,
            "split_policy": args.split_policy,
            "record_order": args.record_order,
            "training_seconds": float(training_seconds),
        }
    )
    return metrics, efficiency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one statistically traceable major-revision experiment."
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--dataset", required=True, choices=["cic_ids2017", "nsl_kdd", "unsw_nb15"]
    )
    parser.add_argument(
        "--model", required=True, choices=sorted(DEEP_MODELS | TREE_MODELS)
    )
    parser.add_argument("--variant", default="full", choices=sorted(VARIANTS))
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument(
        "--leakage-protocol",
        default="clean",
        choices=["clean", "attack_cat", "id", "both"],
    )
    parser.add_argument(
        "--split-policy",
        default="default",
        choices=["default", "ordered", "cic_day_holdout"],
    )
    parser.add_argument(
        "--record-order", default="original", choices=["original", "shuffled"]
    )
    parser.add_argument("--order-seed", type=int, default=2026)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-repetitions", type=int, default=100)

    # Optional one-factor-at-a-time sensitivity overrides.
    parser.add_argument("--hidden-channels", type=int, default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)

    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--tree-learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--tree-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model != "ms_atcn" and args.variant != "full":
        raise ValueError("Only ms_atcn supports ablation variants.")
    if args.dataset != "unsw_nb15" and args.leakage_protocol != "clean":
        raise ValueError("Leakage feature protocols apply only to UNSW-NB15.")
    if args.split_policy == "cic_day_holdout" and args.dataset != "cic_ids2017":
        raise ValueError("cic_day_holdout applies only to CIC-IDS2017.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = dict(config)
    config["seed"] = int(args.seed)
    config["seq_len"] = int(args.seq_len)
    if args.hidden_channels is not None:
        config["hidden_channels"] = int(args.hidden_channels)
    if args.focal_gamma is not None:
        config["focal_gamma"] = float(args.focal_gamma)
    if args.dropout is not None:
        config["dropout"] = float(args.dropout)
    if args.learning_rate is not None:
        config["learning_rate"] = float(args.learning_rate)

    set_reproducible_seed(args.seed)
    device = pick_device(config.get("device", "auto"), args.cpu)
    run_config = vars(args).copy()
    run_config["resolved_device"] = str(device)
    run_config["formula_consistent_focal_loss"] = True
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "config_used.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (output_dir / "environment.json").write_text(
        json.dumps(environment_info(device), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    prepared = prepare_revision_data(
        dataset=args.dataset,
        data_root=args.data_root,
        output_dir=str(output_dir),
        batch_size=int(config.get("batch_size", 128)),
        val_ratio=float(config.get("val_ratio", 0.15)),
        scaling=str(config.get("feature_scaling", "standard")),
        num_workers=int(config.get("num_workers", 0)),
        seed=int(args.seed),
        weighted_sampling=bool(config.get("weighted_sampling", True)),
        seq_len=int(args.seq_len),
        window_stride=int(args.window_stride),
        leakage_protocol=args.leakage_protocol,
        split_policy=args.split_policy,
        record_order=args.record_order,
        order_seed=int(args.order_seed),
    )
    print(json.dumps(prepared.data_info, indent=2, ensure_ascii=False), flush=True)

    if args.model in DEEP_MODELS:
        metrics, efficiency = train_deep(
            args, config, prepared, output_dir, device
        )
    else:
        metrics, efficiency = train_tree(args, prepared, output_dir)

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "efficiency.json").write_text(
        json.dumps(efficiency, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("FINAL_METRICS")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
