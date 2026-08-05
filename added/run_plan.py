from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

ALL_MODELS = ["ms_atcn", "mlp", "cnn", "lstm", "gru", "tcn", "xgboost", "lightgbm"]
DEFAULT_MAIN_MODELS = {
    # Pre-specified from the submitted manuscript: strongest neural comparator
    # for each dataset, plus both strong tree baselines.
    "cic_ids2017": ["ms_atcn", "mlp", "xgboost", "lightgbm"],
    "nsl_kdd": ["ms_atcn", "gru", "xgboost", "lightgbm"],
    "unsw_nb15": ["ms_atcn", "cnn", "xgboost", "lightgbm"],
}


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--root", default="results/revision_required")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-models", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")


def run_one(
    command: list[str],
    output_dir: Path,
    log_path: Path,
    force: bool,
    dry_run: bool,
) -> None:
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not force:
        print(f"[SKIP] {metrics_path}")
        return
    print("[RUN]", " ".join(command))
    if dry_run:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        tail = ""
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-40:])
        except Exception:
            pass
        raise RuntimeError(
            f"Experiment failed with code {completed.returncode}. Log: {log_path}\n{tail}"
        )
    print(f"[OK] {output_dir}")


def train_command(
    args: argparse.Namespace,
    dataset: str,
    model: str,
    output_dir: Path,
    seed: int,
    *,
    variant: str = "full",
    seq_len: int = 16,
    leakage_protocol: str = "clean",
    split_policy: str = "default",
    record_order: str = "original",
    hidden_channels: int | None = None,
    focal_gamma: float | None = None,
    dropout: float | None = None,
    learning_rate: float | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "revision_required.train_one",
        "--config",
        args.config,
        "--dataset",
        dataset,
        "--model",
        model,
        "--variant",
        variant,
        "--data-root",
        args.data_root,
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--seq-len",
        str(seq_len),
        "--leakage-protocol",
        leakage_protocol,
        "--split-policy",
        split_policy,
        "--record-order",
        record_order,
    ]
    if hidden_channels is not None:
        cmd.extend(["--hidden-channels", str(hidden_channels)])
    if focal_gamma is not None:
        cmd.extend(["--focal-gamma", str(focal_gamma)])
    if dropout is not None:
        cmd.extend(["--dropout", str(dropout)])
    if learning_rate is not None:
        cmd.extend(["--learning-rate", str(learning_rate)])
    if args.cpu:
        cmd.append("--cpu")
    if not args.keep_models:
        cmd.append("--no-save-model")
    if args.save_predictions:
        cmd.append("--save-predictions")
    return cmd


def run_main(args: argparse.Namespace) -> None:
    root = Path(args.root)
    first_seed = int(args.seeds[0])
    for dataset in args.datasets:
        # Re-run the complete eight-model table once under the corrected pipeline.
        for model in ALL_MODELS:
            out = root / "main" / dataset / model / f"seed_{first_seed}"
            log = root / "logs" / "main" / dataset / f"{model}_seed_{first_seed}.log"
            run_one(
                train_command(args, dataset, model, out, first_seed),
                out,
                log,
                args.force,
                args.dry_run,
            )

        # Add remaining seeds for the pre-specified statistical comparison set,
        # or for every model when --all-models-all-seeds is requested.
        repeated_models = ALL_MODELS if args.all_models_all_seeds else DEFAULT_MAIN_MODELS[dataset]
        for model in repeated_models:
            for seed in args.seeds[1:]:
                out = root / "main" / dataset / model / f"seed_{seed}"
                log = root / "logs" / "main" / dataset / f"{model}_seed_{seed}.log"
                run_one(
                    train_command(args, dataset, model, out, seed),
                    out,
                    log,
                    args.force,
                    args.dry_run,
                )


def run_ablation(args: argparse.Namespace) -> None:
    root = Path(args.root)
    # Full MS-ATCN is reused from main/<dataset>/ms_atcn/seed_<seed>.
    for dataset in args.datasets:
        for variant in ["no_multiscale", "no_attention", "no_focal_loss"]:
            for seed in args.seeds:
                out = root / "ablation" / dataset / variant / f"seed_{seed}"
                log = root / "logs" / "ablation" / dataset / f"{variant}_seed_{seed}.log"
                run_one(
                    train_command(
                        args,
                        dataset,
                        "ms_atcn",
                        out,
                        seed,
                        variant=variant,
                    ),
                    out,
                    log,
                    args.force,
                    args.dry_run,
                )


def run_leakage(args: argparse.Namespace) -> None:
    root = Path(args.root)
    for model in ["ms_atcn", "lightgbm"]:
        for seed in args.seeds:
            main_clean = root / "main" / "unsw_nb15" / model / f"seed_{seed}" / "metrics.json"
            if not main_clean.exists():
                out = root / "leakage_unsw" / "clean" / model / f"seed_{seed}"
                log = root / "logs" / "leakage_unsw" / f"clean_{model}_seed_{seed}.log"
                run_one(
                    train_command(
                        args,
                        "unsw_nb15",
                        model,
                        out,
                        seed,
                        leakage_protocol="clean",
                    ),
                    out,
                    log,
                    args.force,
                    args.dry_run,
                )
        for protocol in ["attack_cat", "id", "both"]:
            for seed in args.seeds:
                out = root / "leakage_unsw" / protocol / model / f"seed_{seed}"
                log = (
                    root
                    / "logs"
                    / "leakage_unsw"
                    / f"{protocol}_{model}_seed_{seed}.log"
                )
                run_one(
                    train_command(
                        args,
                        "unsw_nb15",
                        model,
                        out,
                        seed,
                        leakage_protocol=protocol,
                    ),
                    out,
                    log,
                    args.force,
                    args.dry_run,
                )


def run_window(args: argparse.Namespace) -> None:
    root = Path(args.root)
    for dataset in args.datasets:
        for seq_len in args.window_lengths:
            for seed in args.seeds:
                if seq_len == 16:
                    main_ref = (
                        root / "main" / dataset / "ms_atcn" / f"seed_{seed}" / "metrics.json"
                    )
                    if main_ref.exists():
                        print(f"[REUSE] {main_ref}")
                        continue
                out = root / "window" / dataset / f"L{seq_len}" / f"seed_{seed}"
                log = (
                    root
                    / "logs"
                    / "window"
                    / dataset
                    / f"L{seq_len}_seed_{seed}.log"
                )
                run_one(
                    train_command(
                        args,
                        dataset,
                        "ms_atcn",
                        out,
                        seed,
                        seq_len=seq_len,
                    ),
                    out,
                    log,
                    args.force,
                    args.dry_run,
                )


def run_order(args: argparse.Namespace) -> None:
    root = Path(args.root)
    for dataset in args.datasets:
        for seed in args.seeds:
            main_ref = root / "main" / dataset / "ms_atcn" / f"seed_{seed}" / "metrics.json"
            if not main_ref.exists():
                out = root / "order_sanity" / dataset / "original" / f"seed_{seed}"
                log = (
                    root
                    / "logs"
                    / "order_sanity"
                    / dataset
                    / f"original_seed_{seed}.log"
                )
                run_one(
                    train_command(args, dataset, "ms_atcn", out, seed),
                    out,
                    log,
                    args.force,
                    args.dry_run,
                )
            out = root / "order_sanity" / dataset / "shuffled" / f"seed_{seed}"
            log = (
                root
                / "logs"
                / "order_sanity"
                / dataset
                / f"shuffled_seed_{seed}.log"
            )
            run_one(
                train_command(
                    args,
                    dataset,
                    "ms_atcn",
                    out,
                    seed,
                    record_order="shuffled",
                ),
                out,
                log,
                args.force,
                args.dry_run,
            )


def run_hyper(args: argparse.Namespace) -> None:
    """Compact one-factor-at-a-time sensitivity on the strongest MS-ATCN dataset."""
    root = Path(args.root)
    dataset = "unsw_nb15"
    seed = int(args.seeds[0])
    plans = {
        "hidden_channels": [32, 64, 128],
        "focal_gamma": [1.0, 2.0, 3.0],
        "learning_rate": [0.0001, 0.0005, 0.001],
        "dropout": [0.0, 0.1, 0.3],
    }
    base_values = {
        "hidden_channels": 64,
        "focal_gamma": 2.0,
        "learning_rate": 0.0005,
        "dropout": 0.1,
    }
    for parameter, values in plans.items():
        for value in values:
            if float(value) == float(base_values[parameter]):
                main_ref = root / "main" / dataset / "ms_atcn" / f"seed_{seed}" / "metrics.json"
                if main_ref.exists():
                    print(f"[REUSE] {parameter}={value}: {main_ref}")
                    continue
            token = str(value).replace(".", "p")
            out = root / "hyper_unsw" / parameter / token / f"seed_{seed}"
            log = root / "logs" / "hyper_unsw" / parameter / f"{token}_seed_{seed}.log"
            overrides = {
                "hidden_channels": None,
                "focal_gamma": None,
                "dropout": None,
                "learning_rate": None,
            }
            overrides[parameter] = value
            run_one(
                train_command(
                    args,
                    dataset,
                    "ms_atcn",
                    out,
                    seed,
                    **overrides,
                ),
                out,
                log,
                args.force,
                args.dry_run,
            )


def run_temporal(args: argparse.Namespace) -> None:
    root = Path(args.root)
    for model in ["ms_atcn", "mlp", "xgboost", "lightgbm"]:
        for seed in args.seeds:
            out = root / "temporal_cic" / model / f"seed_{seed}"
            log = root / "logs" / "temporal_cic" / f"{model}_seed_{seed}.log"
            run_one(
                train_command(
                    args,
                    "cic_ids2017",
                    model,
                    out,
                    seed,
                    split_policy="cic_day_holdout",
                ),
                out,
                log,
                args.force,
                args.dry_run,
            )


def run_audit(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "revision_required.audit_leakage",
        "--data-root",
        args.data_root,
        "--output-dir",
        str(Path(args.root) / "audit"),
        "--include-cic-day-holdout",
    ]
    if args.datasets:
        cmd.extend(["--datasets", *args.datasets])
    print("[RUN]", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def run_aggregate(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "revision_required.aggregate_results",
        "--root",
        args.root,
        "--output-dir",
        str(Path(args.root) / "summary"),
    ]
    print("[RUN]", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the experiments needed for the major revision."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="No-training leakage and duplicate audit")
    add_common(audit)
    audit.add_argument(
        "--datasets",
        nargs="+",
        default=["cic_ids2017", "nsl_kdd", "unsw_nb15"],
        choices=["cic_ids2017", "nsl_kdd", "unsw_nb15"],
    )

    main = sub.add_parser("main", help="Six-seed main comparison")
    add_common(main)
    main.add_argument(
        "--datasets",
        nargs="+",
        default=["cic_ids2017", "nsl_kdd", "unsw_nb15"],
        choices=["cic_ids2017", "nsl_kdd", "unsw_nb15"],
    )
    main.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46, 47])
    main.add_argument(
        "--all-models-all-seeds",
        action="store_true",
        help="Repeat all eight models for every seed instead of the compressed comparison set.",
    )

    ablation = sub.add_parser("ablation", help="Six-seed component ablation")
    add_common(ablation)
    ablation.add_argument(
        "--datasets",
        nargs="+",
        default=["cic_ids2017", "nsl_kdd", "unsw_nb15"],
        choices=["cic_ids2017", "nsl_kdd", "unsw_nb15"],
    )
    ablation.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46, 47])

    leakage = sub.add_parser("leakage", help="UNSW leakage before/after quantification")
    add_common(leakage)
    leakage.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])

    window = sub.add_parser("window", help="Window length sensitivity")
    add_common(window)
    window.add_argument(
        "--datasets",
        nargs="+",
        default=["cic_ids2017", "nsl_kdd", "unsw_nb15"],
        choices=["cic_ids2017", "nsl_kdd", "unsw_nb15"],
    )
    window.add_argument("--seeds", nargs="+", type=int, default=[42])
    window.add_argument(
        "--window-lengths", nargs="+", type=int, default=[1, 8, 16, 32]
    )

    order = sub.add_parser("order", help="Original versus shuffled record-order sanity check")
    add_common(order)
    order.add_argument(
        "--datasets",
        nargs="+",
        default=["cic_ids2017", "nsl_kdd", "unsw_nb15"],
        choices=["cic_ids2017", "nsl_kdd", "unsw_nb15"],
    )
    order.add_argument("--seeds", nargs="+", type=int, default=[42])

    hyper = sub.add_parser("hyper", help="Compact principal-hyperparameter sensitivity on UNSW-NB15")
    add_common(hyper)
    hyper.add_argument("--seeds", nargs="+", type=int, default=[42])

    temporal = sub.add_parser("temporal", help="CIC Monday-Wednesday/Thursday/Friday holdout")
    add_common(temporal)
    temporal.add_argument("--seeds", nargs="+", type=int, default=[42])

    aggregate = sub.add_parser("aggregate", help="Create means, standard deviations, and tests")
    add_common(aggregate)

    core = sub.add_parser("core", help="Run audit, main, ablation, leakage, window, hyper, order, temporal, aggregate")
    add_common(core)
    core.add_argument(
        "--datasets",
        nargs="+",
        default=["cic_ids2017", "nsl_kdd", "unsw_nb15"],
        choices=["cic_ids2017", "nsl_kdd", "unsw_nb15"],
    )
    core.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46, 47])
    core.add_argument(
        "--all-models-all-seeds",
        action="store_true",
        help="Repeat all eight main-table models for all seeds.",
    )
    core.add_argument("--window-lengths", nargs="+", type=int, default=[1, 8, 16, 32])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit":
        run_audit(args)
    elif args.command == "main":
        run_main(args)
    elif args.command == "ablation":
        run_ablation(args)
    elif args.command == "leakage":
        run_leakage(args)
    elif args.command == "window":
        run_window(args)
    elif args.command == "order":
        run_order(args)
    elif args.command == "hyper":
        run_hyper(args)
    elif args.command == "temporal":
        run_temporal(args)
    elif args.command == "aggregate":
        run_aggregate(args)
    elif args.command == "core":
        # Use six seeds for statistical main/ablation. Six is the smallest paired
        # sample size for which an exact two-sided Wilcoxon test can attain p < 0.05.
        # Use the first three for
        # sensitivity and realism checks to limit the added compute.
        run_audit(args)
        run_main(args)
        run_ablation(args)

        original_seeds = args.seeds
        args.seeds = original_seeds[:3]
        run_leakage(args)
        args.seeds = original_seeds[:1]
        run_window(args)
        run_hyper(args)
        run_order(args)
        run_temporal(args)
        args.seeds = original_seeds
        run_aggregate(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
