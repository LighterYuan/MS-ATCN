import json
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Basic configuration
# =========================

RESULTS_ROOT = Path("results")

DATASET_DIRS = {
    "CIC-IDS2017": RESULTS_ROOT / "cic_binary",
    "NSL-KDD": RESULTS_ROOT / "nsl_binary",
    "UNSW-NB15": RESULTS_ROOT / "unsw_binary_fix",
}

BASELINE_MODELS = [
    "ms_atcn",
    "mlp",
    "cnn",
    "lstm",
    "gru",
    "tcn",
    "xgboost",
    "lightgbm",
]

BASELINE_DISPLAY_NAMES = {
    "ms_atcn": "MS-ATCN",
    "mlp": "MLP",
    "cnn": "CNN",
    "lstm": "LSTM",
    "gru": "GRU",
    "tcn": "TCN",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

ABLATION_VARIANTS = [
    "no_multiscale",
    "no_attention",
    "no_focal_loss",
    "full_ms_atcn",
]

ABLATION_DISPLAY_NAMES = {
    "no_multiscale": "w/o Multi-scale",
    "no_attention": "w/o Attention",
    "no_focal_loss": "w/o Focal Loss",
    "full_ms_atcn": "Full MS-ATCN",
}


# =========================
# Utility functions
# =========================

def load_metrics_json(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_macro_f1(metrics: dict) -> float:
    """
    Robustly extract Macro-F1 from metrics.json.

    In the current experimental outputs, Macro-F1 is stored as:
        f1_macro
    """
    candidate_keys = [
        "f1_macro",
        "macro_f1",
        "macro_f1_score",
        "macro avg/f1-score",
        "macro_avg_f1",
        "macro_f1_test",
    ]

    for key in candidate_keys:
        if key in metrics:
            return float(metrics[key])

    # Some metrics.json may store nested classification report-like objects.
    if "macro avg" in metrics and isinstance(metrics["macro avg"], dict):
        macro_avg = metrics["macro avg"]
        if "f1-score" in macro_avg:
            return float(macro_avg["f1-score"])

    raise KeyError(
        "Cannot find Macro-F1 in metrics.json. "
        f"Available keys: {list(metrics.keys())}"
    )


def find_metrics_file(base_dir: Path, name: str) -> Path:
    """
    Try several possible directory layouts.

    Supported examples:
    1. results/cic_binary/ms_atcn/metrics.json
    2. results/cic_binary/ms_atcn_fixed/metrics.json
    3. results/cic_binary/baselines/mlp/metrics.json
    4. results/cic_binary/ablation/no_attention/metrics.json
    """
    candidate_paths = [
        base_dir / name / "metrics.json",
        base_dir / f"{name}_fixed" / "metrics.json",
        base_dir / "baselines" / name / "metrics.json",
        base_dir / "baseline" / name / "metrics.json",
        base_dir / "ablation" / name / "metrics.json",
        base_dir / "ablations" / name / "metrics.json",
        base_dir / "ablation" / f"{name}_fixed" / "metrics.json",
        base_dir / "ablations" / f"{name}_fixed" / "metrics.json",
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    # Fallback: recursive search.
    # This helps if your directory names contain extra suffixes.
    matches = list(base_dir.rglob(f"*{name}*/metrics.json"))

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        # Prefer non-DR and non-temporary results if several matches exist.
        filtered = [
            p for p in matches
            if "dr" not in str(p).lower()
            and "decision" not in str(p).lower()
            and "tmp" not in str(p).lower()
        ]
        if len(filtered) == 1:
            return filtered[0]

        raise RuntimeError(
            f"Multiple metrics.json files found for '{name}' under {base_dir}:\n"
            + "\n".join(str(p) for p in matches)
            + "\nPlease make the directory names unambiguous or edit find_metrics_file()."
        )

    raise FileNotFoundError(
        f"No metrics.json found for '{name}' under {base_dir}."
    )


def save_csv(path: Path, header: list, rows: list):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# =========================
# Figure 2
# =========================

def collect_baseline_macro_f1():
    values = {}

    for dataset_name, dataset_dir in DATASET_DIRS.items():
        values[dataset_name] = []

        for model in BASELINE_MODELS:
            metrics_path = find_metrics_file(dataset_dir, model)
            metrics = load_metrics_json(metrics_path)
            macro_f1 = get_macro_f1(metrics)

            values[dataset_name].append(macro_f1)

            print(
                f"[Figure 2] {dataset_name:12s} | "
                f"{BASELINE_DISPLAY_NAMES[model]:8s} | "
                f"Macro-F1 = {macro_f1:.4f} | {metrics_path}"
            )

    return values


def plot_fig2(values):
    datasets = list(DATASET_DIRS.keys())
    models = BASELINE_MODELS
    display_models = [BASELINE_DISPLAY_NAMES[m] for m in models]

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(10, 5))

    for i, dataset in enumerate(datasets):
        plt.bar(
            x + (i - 1) * width,
            values[dataset],
            width,
            label=dataset,
        )

    plt.ylabel("Macro-F1")
    plt.xlabel("Model")
    plt.title("Cross-dataset Macro-F1 Comparison of Baseline Models")
    plt.xticks(x, display_models, rotation=30, ha="right")
    plt.ylim(0.70, 1.00)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig2_macro_f1_comparison.png", dpi=300)
    plt.close()

    rows = []
    for dataset in datasets:
        for model, value in zip(display_models, values[dataset]):
            rows.append([dataset, model, f"{value:.4f}"])

    save_csv(
        Path("figure2_macro_f1_values.csv"),
        ["Dataset", "Model", "Macro-F1"],
        rows,
    )


# =========================
# Figure 3
# =========================

def collect_ablation_macro_f1():
    values = {}

    for dataset_name, dataset_dir in DATASET_DIRS.items():
        values[dataset_name] = []

        for variant in ABLATION_VARIANTS:
            metrics_path = find_metrics_file(dataset_dir, variant)
            metrics = load_metrics_json(metrics_path)
            macro_f1 = get_macro_f1(metrics)

            values[dataset_name].append(macro_f1)

            print(
                f"[Figure 3] {dataset_name:12s} | "
                f"{ABLATION_DISPLAY_NAMES[variant]:16s} | "
                f"Macro-F1 = {macro_f1:.4f} | {metrics_path}"
            )

    return values


def plot_fig3(values):
    datasets = list(DATASET_DIRS.keys())
    variants = ABLATION_VARIANTS
    display_variants = [ABLATION_DISPLAY_NAMES[v] for v in variants]

    x = np.arange(len(variants))
    width = 0.25

    plt.figure(figsize=(8, 5))

    for i, dataset in enumerate(datasets):
        plt.bar(
            x + (i - 1) * width,
            values[dataset],
            width,
            label=dataset,
        )

    plt.ylabel("Macro-F1")
    plt.xlabel("Ablation Variant")
    plt.title("Macro-F1 Comparison of MS-ATCN Ablation Variants")
    plt.xticks(x, display_variants, rotation=20, ha="right")
    plt.ylim(0.70, 0.90)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig3_ablation_macro_f1.png", dpi=300)
    plt.close()

    rows = []
    for dataset in datasets:
        for variant, value in zip(display_variants, values[dataset]):
            rows.append([dataset, variant, f"{value:.4f}"])

    save_csv(
        Path("figure3_ablation_macro_f1_values.csv"),
        ["Dataset", "Variant", "Macro-F1"],
        rows,
    )


# =========================
# Main
# =========================

def main():
    print("Reading metrics.json files from:", RESULTS_ROOT.resolve())

    print("\nCollecting baseline Macro-F1 values...")
    baseline_values = collect_baseline_macro_f1()
    plot_fig2(baseline_values)

    print("\nCollecting ablation Macro-F1 values...")
    ablation_values = collect_ablation_macro_f1()
    plot_fig3(ablation_values)

    print("\nDone. Generated files:")
    print("  fig2_macro_f1_comparison.png")
    print("  fig3_ablation_macro_f1.png")
    print("  figure2_macro_f1_values.csv")
    print("  figure3_ablation_macro_f1_values.csv")


if __name__ == "__main__":
    main()
