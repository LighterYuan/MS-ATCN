from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon

METRICS = ["accuracy", "f1_macro", "auc", "fpr_macro", "fpr_attack_positive"]


def read_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    data["_path"] = str(path)
    return data


def collect_metrics(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.rglob("metrics.json"):
        data = read_metrics(path)
        if data is None:
            continue
        rel = path.relative_to(root)
        data["relative_path"] = str(rel)
        data["family"] = rel.parts[0] if rel.parts else "unknown"
        rows.append(data)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for keys, group in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        base["n_runs"] = int(len(group))
        for metric in METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            if len(values) == 0:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ci = (
                float(t.ppf(0.975, df=len(values) - 1) * std / math.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
            base[f"{metric}_mean"] = mean
            base[f"{metric}_std"] = std
            base[f"{metric}_ci95_halfwidth"] = ci
            base[f"{metric}_mean_std"] = f"{mean:.4f} ± {std:.4f}"
        rows.append(base)
    return pd.DataFrame(rows)


def paired_wilcoxon(
    a: pd.DataFrame,
    b: pd.DataFrame,
    metric: str,
    a_name: str,
    b_name: str,
    context: Dict[str, object],
) -> dict | None:
    aa = a[["seed", metric]].copy()
    bb = b[["seed", metric]].copy()
    aa["seed"] = pd.to_numeric(aa["seed"], errors="coerce")
    bb["seed"] = pd.to_numeric(bb["seed"], errors="coerce")
    merged = aa.merge(bb, on="seed", suffixes=("_a", "_b")).dropna()
    if len(merged) < 2:
        return None
    x = merged[f"{metric}_a"].to_numpy(float)
    y = merged[f"{metric}_b"].to_numpy(float)
    differences = x - y
    if np.allclose(differences, 0):
        statistic, pvalue = 0.0, 1.0
    else:
        try:
            statistic, pvalue = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            statistic, pvalue = np.nan, 1.0
    return {
        **context,
        "metric": metric,
        "comparison_a": a_name,
        "comparison_b": b_name,
        "n_pairs": int(len(merged)),
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
        "mean_difference_a_minus_b": float(differences.mean()),
        "median_difference_a_minus_b": float(np.median(differences)),
        "wins_a": int(np.sum(differences > 0)),
        "ties": int(np.sum(np.isclose(differences, 0))),
        "wins_b": int(np.sum(differences < 0)),
        "exact_two_sided_can_reach_0_05": bool(len(merged) >= 6),
        "wilcoxon_statistic": float(statistic),
        "p_value": float(pvalue),
    }


def holm_adjust(frame: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["p_holm"] = np.nan
    frame["significant_0_05_holm"] = False
    for _, idx in frame.groupby(list(group_cols), dropna=False).groups.items():
        idx = list(idx)
        p = frame.loc[idx, "p_value"].to_numpy(float)
        order = np.argsort(p)
        adjusted = np.empty_like(p)
        running = 0.0
        m = len(p)
        for rank, position in enumerate(order):
            value = min(1.0, (m - rank) * p[position])
            running = max(running, value)
            adjusted[position] = running
        frame.loc[idx, "p_holm"] = adjusted
        frame.loc[idx, "significant_0_05_holm"] = adjusted < 0.05
    return frame


def _main_table(df: pd.DataFrame) -> pd.DataFrame:
    main = df[df["family"] == "main"].copy()
    return summarize(main, ["dataset", "model", "variant"])


def _ablation_table_and_tests(df: pd.DataFrame):
    main_full = df[(df["family"] == "main") & (df["model"] == "ms_atcn")].copy()
    abl = df[df["family"] == "ablation"].copy()
    combined = pd.concat(
        [
            main_full.assign(ablation_variant="full"),
            abl.assign(ablation_variant=abl["variant"]),
        ],
        ignore_index=True,
    )
    summary = summarize(combined, ["dataset", "ablation_variant"])
    tests = []
    for dataset in sorted(combined["dataset"].dropna().unique()):
        full = combined[
            (combined["dataset"] == dataset)
            & (combined["ablation_variant"] == "full")
        ]
        for variant in ["no_multiscale", "no_attention", "no_focal_loss"]:
            other = combined[
                (combined["dataset"] == dataset)
                & (combined["ablation_variant"] == variant)
            ]
            for metric in ["accuracy", "f1_macro", "auc"]:
                row = paired_wilcoxon(
                    full,
                    other,
                    metric,
                    "full",
                    variant,
                    {"family": "ablation", "dataset": dataset},
                )
                if row:
                    tests.append(row)
    test_df = holm_adjust(pd.DataFrame(tests), ["family", "dataset", "metric"])
    return summary, test_df


def _main_tests(df: pd.DataFrame) -> pd.DataFrame:
    main = df[df["family"] == "main"].copy()
    rows = []
    for dataset in sorted(main["dataset"].dropna().unique()):
        full = main[(main["dataset"] == dataset) & (main["model"] == "ms_atcn")]
        for model in sorted(main.loc[main["dataset"] == dataset, "model"].dropna().unique()):
            if model == "ms_atcn":
                continue
            other = main[(main["dataset"] == dataset) & (main["model"] == model)]
            for metric in ["accuracy", "f1_macro", "auc"]:
                row = paired_wilcoxon(
                    full,
                    other,
                    metric,
                    "ms_atcn",
                    model,
                    {"family": "main", "dataset": dataset},
                )
                if row:
                    rows.append(row)
    return holm_adjust(pd.DataFrame(rows), ["family", "dataset", "metric"])


def _add_reference_rows(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows = [df]
    main = df[df["family"] == "main"].copy()

    # Clean UNSW reference for leakage experiment, only when no explicit run exists.
    clean = main[
        (main["dataset"] == "unsw_nb15")
        & (main["model"].isin(["ms_atcn", "lightgbm"]))
    ].copy()
    explicit_leak = df[df["family"] == "leakage_unsw"]
    if not clean.empty:
        keep = []
        explicit_keys = set(
            zip(
                explicit_leak.get("model", pd.Series(dtype=str)).astype(str),
                pd.to_numeric(explicit_leak.get("seed", pd.Series(dtype=float)), errors="coerce"),
                explicit_leak.get("leakage_protocol", pd.Series(dtype=str)).astype(str),
            )
        )
        for idx, row in clean.iterrows():
            key = (str(row.get("model")), float(row.get("seed")), "clean")
            keep.append(key not in explicit_keys)
        clean = clean.loc[keep].copy()
        clean["family"] = "leakage_unsw"
        clean["leakage_protocol"] = "clean"
        rows.append(clean)

    # L=16 references for window sensitivity, only when no explicit run exists.
    w16 = main[main["model"] == "ms_atcn"].copy()
    explicit_window = df[df["family"] == "window"]
    if not w16.empty:
        explicit_keys = set(
            zip(
                explicit_window.get("dataset", pd.Series(dtype=str)).astype(str),
                pd.to_numeric(explicit_window.get("seed", pd.Series(dtype=float)), errors="coerce"),
                pd.to_numeric(explicit_window.get("seq_len", pd.Series(dtype=float)), errors="coerce"),
            )
        )
        keep = [
            (str(row.get("dataset")), float(row.get("seed")), 16.0) not in explicit_keys
            for _, row in w16.iterrows()
        ]
        w16 = w16.loc[keep].copy()
        w16["family"] = "window"
        w16["seq_len"] = 16
        rows.append(w16)

    # Base-value references for one-factor-at-a-time hyperparameter sensitivity.
    hyper_explicit = df[df["family"] == "hyper_unsw"]
    hyper_base = main[
        (main["dataset"] == "unsw_nb15") & (main["model"] == "ms_atcn")
    ].copy()
    base_tokens = {
        "hidden_channels": "64",
        "focal_gamma": "2p0",
        "learning_rate": "0p0005",
        "dropout": "0p1",
    }
    explicit_paths = set(hyper_explicit.get("relative_path", pd.Series(dtype=str)).astype(str))
    for parameter, token in base_tokens.items():
        ref = hyper_base.copy()
        if ref.empty:
            continue
        ref["family"] = "hyper_unsw"
        ref["relative_path"] = ref["seed"].map(
            lambda seed: f"hyper_unsw/{parameter}/{token}/seed_{int(seed)}/metrics.json"
        )
        ref = ref[~ref["relative_path"].isin(explicit_paths)]
        rows.append(ref)

    # Original-order references, only when no explicit run exists.
    original = main[main["model"] == "ms_atcn"].copy()
    explicit_order = df[df["family"] == "order_sanity"]
    if not original.empty:
        explicit_keys = set(
            zip(
                explicit_order.get("dataset", pd.Series(dtype=str)).astype(str),
                pd.to_numeric(explicit_order.get("seed", pd.Series(dtype=float)), errors="coerce"),
                explicit_order.get("record_order", pd.Series(dtype=str)).astype(str),
            )
        )
        keep = [
            (str(row.get("dataset")), float(row.get("seed")), "original") not in explicit_keys
            for _, row in original.iterrows()
        ]
        original = original.loc[keep].copy()
        original["family"] = "order_sanity"
        original["record_order"] = "original"
        rows.append(original)
    return pd.concat(rows, ignore_index=True)


def aggregate(root: Path, output_dir: Path) -> None:
    raw = collect_metrics(root)
    if raw.empty:
        raise FileNotFoundError(f"No metrics.json files found under {root}")
    raw = _add_reference_rows(raw, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "all_run_metrics.csv", index=False)

    main_summary = _main_table(raw)
    main_summary.to_csv(output_dir / "main_multiseed_summary.csv", index=False)

    main_tests = _main_tests(raw)
    main_tests.to_csv(output_dir / "main_pairwise_wilcoxon.csv", index=False)

    ablation_summary, ablation_tests = _ablation_table_and_tests(raw)
    ablation_summary.to_csv(output_dir / "ablation_summary.csv", index=False)
    ablation_tests.to_csv(output_dir / "ablation_wilcoxon_holm.csv", index=False)

    leakage = raw[raw["family"] == "leakage_unsw"].copy()
    leakage_summary = summarize(leakage, ["leakage_protocol", "model"])
    leakage_summary.to_csv(output_dir / "unsw_leakage_summary.csv", index=False)
    leak_tests = []
    for model in sorted(leakage["model"].dropna().unique()):
        clean = leakage[
            (leakage["model"] == model) & (leakage["leakage_protocol"] == "clean")
        ]
        for protocol in ["attack_cat", "id", "both"]:
            leaky = leakage[
                (leakage["model"] == model)
                & (leakage["leakage_protocol"] == protocol)
            ]
            for metric in ["accuracy", "f1_macro", "auc"]:
                row = paired_wilcoxon(
                    leaky,
                    clean,
                    metric,
                    protocol,
                    "clean",
                    {"family": "leakage_unsw", "model": model},
                )
                if row:
                    leak_tests.append(row)
    holm_adjust(
        pd.DataFrame(leak_tests), ["family", "model", "metric"]
    ).to_csv(output_dir / "unsw_leakage_wilcoxon_holm.csv", index=False)

    window = raw[raw["family"] == "window"].copy()
    summarize(window, ["dataset", "seq_len"]).to_csv(
        output_dir / "window_sensitivity_summary.csv", index=False
    )

    order = raw[raw["family"] == "order_sanity"].copy()
    summarize(order, ["dataset", "record_order"]).to_csv(
        output_dir / "order_sanity_summary.csv", index=False
    )
    order_tests = []
    for dataset in sorted(order["dataset"].dropna().unique()):
        original = order[
            (order["dataset"] == dataset) & (order["record_order"] == "original")
        ]
        shuffled = order[
            (order["dataset"] == dataset) & (order["record_order"] == "shuffled")
        ]
        for metric in ["accuracy", "f1_macro", "auc"]:
            row = paired_wilcoxon(
                original,
                shuffled,
                metric,
                "original",
                "shuffled",
                {"family": "order_sanity", "dataset": dataset},
            )
            if row:
                order_tests.append(row)
    holm_adjust(
        pd.DataFrame(order_tests), ["family", "dataset", "metric"]
    ).to_csv(output_dir / "order_sanity_wilcoxon_holm.csv", index=False)

    temporal = raw[raw["family"] == "temporal_cic"].copy()
    summarize(temporal, ["dataset", "model"]).to_csv(
        output_dir / "cic_temporal_summary.csv", index=False
    )

    hyper = raw[raw["family"] == "hyper_unsw"].copy()
    if not hyper.empty:
        hyper["sensitivity_parameter"] = hyper["relative_path"].map(
            lambda x: Path(str(x)).parts[1] if len(Path(str(x)).parts) > 1 else None
        )
        hyper["sensitivity_value"] = hyper["relative_path"].map(
            lambda x: Path(str(x)).parts[2] if len(Path(str(x)).parts) > 2 else None
        )
        summarize(hyper, ["sensitivity_parameter", "sensitivity_value"]).to_csv(
            output_dir / "hyperparameter_sensitivity_summary.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output_dir / "hyperparameter_sensitivity_summary.csv", index=False)

    efficiency_rows = []
    for path in root.rglob("efficiency.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["relative_path"] = str(path.relative_to(root))
        payload["family"] = path.relative_to(root).parts[0]
        inference = payload.pop("inference", None)
        if isinstance(inference, dict):
            for key, value in inference.items():
                if isinstance(value, dict):
                    for metric, val in value.items():
                        payload[f"inference_{key}_{metric}"] = val
                else:
                    payload[f"inference_{key}"] = value
        efficiency_rows.append(payload)
    pd.DataFrame(efficiency_rows).to_csv(
        output_dir / "efficiency_all_runs.csv", index=False
    )

    readme = f"""# Major-revision result summary

Generated from: `{root}`

Core tables:
- `main_multiseed_summary.csv`
- `main_pairwise_wilcoxon.csv`
- `ablation_summary.csv`
- `ablation_wilcoxon_holm.csv`
- `unsw_leakage_summary.csv`
- `unsw_leakage_wilcoxon_holm.csv`
- `window_sensitivity_summary.csv`
- `order_sanity_summary.csv`
- `order_sanity_wilcoxon_holm.csv`
- `cic_temporal_summary.csv`
- `hyperparameter_sensitivity_summary.csv`
- `efficiency_all_runs.csv`

All p-values are two-sided paired Wilcoxon signed-rank tests using seed-aligned runs.
Holm correction is applied within each dataset/model/metric comparison family.
With six seeds, statistical power remains limited; report effect sizes and mean ± standard deviation
alongside p-values and avoid universal superiority claims.
"""
    (output_dir / "README_SUMMARY.md").write_text(readme, encoding="utf-8")
    print(f"Saved aggregate outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate required revision experiments.")
    parser.add_argument("--root", default="results/revision_required")
    parser.add_argument(
        "--output-dir", default="results/revision_required/summary"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate(Path(args.root), Path(args.output_dir))


if __name__ == "__main__":
    main()
