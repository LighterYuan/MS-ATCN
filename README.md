# MS-ATCN-IDS: Reproducible Intrusion Detection Project

A reproducible PyTorch project for **network intrusion detection** based on a **Multi-Scale Attention Temporal Convolutional Network (MS-ATCN)** with optional **Focal Loss** for class imbalance.

This project is designed to support a paper built on improving temporal convolutional models for intrusion detection.

## Features

- Reproducible training with fixed seeds
- CSV-based data pipeline for IDS datasets
- Automatic train/val/test split
- Standardization and label encoding
- Multi-scale dilated temporal convolutions
- Channel attention and temporal attention
- Optional Focal Loss / weighted cross entropy
- Metrics: Accuracy, Precision, Recall, F1, Macro-F1, Weighted-F1, FPR, confusion matrix
- Ablation-friendly switches
- Synthetic demo dataset generator for smoke testing

## Recommended datasets

You can adapt this code to datasets such as:

- CSE-CIC-IDS2018
- CICDDoS2019
- CIC-IDS2017
- UNSW-NB15
- NSL-KDD

## Project structure

```text
ids_msatcn_project/
  configs/
    base.yaml
  scripts/
    make_synthetic_ids.py
  src/
    data.py
    losses.py
    metrics.py
    model.py
    train.py
  results/
  requirements.txt
  README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start with synthetic data

Generate a synthetic multi-class IDS-like dataset:

```bash
python scripts/make_synthetic_ids.py --output data_synth.csv --samples 4000 --features 32 --classes 6
```

Train the model:

```bash
python -m src.train --config configs/base.yaml --csv_path data_synth.csv --label_col label --output_dir results/demo_run
```

## Running on your own dataset

Prepare a CSV file where:

- each row is a sample / flow / record
- feature columns are numeric
- one column is the label column (for example `Label` or `label`)

Example:

```bash
python -m src.train \
  --config configs/base.yaml \
  --csv_path /path/to/your_dataset.csv \
  --label_col Label \
  --output_dir results/cic_ids_run
```

## Paper-ready experiment suggestions

### Main model

- `use_channel_attention: true`
- `use_temporal_attention: true`
- `use_focal_loss: true`
- `multi_scale_kernel_sizes: [3,5,7]`
- `dilations: [1,2,4,8]`

### Baseline / ablation settings

1. Remove attention modules
2. Replace Focal Loss with weighted cross entropy
3. Use single-scale kernel only
4. Reduce dilation depth

## Outputs

Each run saves:

- `config_used.yaml`
- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.csv`
- `training_history.csv`
- `best_model.pt`
- `label_mapping.json`

## Reproducibility

- fixed random seeds
- deterministic PyTorch flags where possible
- saved config and label mapping
- explicit preprocessing pipeline

## Suggested paper title

**A Multi-Scale Attention Temporal Convolutional Network for Network Intrusion Detection Under Class Imbalance**

