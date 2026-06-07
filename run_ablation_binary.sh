#!/usr/bin/env bash
set -e

# Ablation experiments for binary MS-ATCN.
# This script does NOT modify src/train.py, src/model.py, src/data.py, src/metrics.py,
# and does NOT touch any baseline code.
#
# It creates temporary config files from configs/base.yaml and runs:
# 1) no_multiscale
# 2) no_attention
# 3) no_focal_loss
# 4) full_ms_atcn
#
# Output directories are separated from existing MS-ATCN and baseline results.

PROJECT_ROOT="$(pwd)"
BASE_CONFIG="configs/base.yaml"
ABLATION_CONFIG_DIR="configs/ablation_generated"

if [ ! -f "$BASE_CONFIG" ]; then
  echo "ERROR: $BASE_CONFIG not found. Please run this script from the project root."
  exit 1
fi

mkdir -p "$ABLATION_CONFIG_DIR"

python - <<'PY'
from pathlib import Path
import yaml

base_path = Path("configs/base.yaml")
out_dir = Path("configs/ablation_generated")
out_dir.mkdir(parents=True, exist_ok=True)

with base_path.open("r", encoding="utf-8") as f:
    base = yaml.safe_load(f)

def write_variant(name, updates):
    cfg = dict(base)
    cfg.update(updates)
    path = out_dir / f"{name}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"generated {path}")

# w/o Multi-scale:
# keep the same MS-ATCN training pipeline, but replace multi-kernel branches [3,5,7]
# with a single temporal convolution scale [3].
write_variant("no_multiscale", {
    "multi_scale_kernel_sizes": [3],
    "use_channel_attention": True,
    "use_temporal_attention": True,
    "use_focal_loss": True,
})

# w/o Attention:
# disable both channel attention and temporal attention.
write_variant("no_attention", {
    "use_channel_attention": False,
    "use_temporal_attention": False,
    "use_focal_loss": True,
})

# w/o Focal Loss:
# replace Focal Loss with weighted CrossEntropyLoss in src/train.py.
write_variant("no_focal_loss", {
    "use_channel_attention": True,
    "use_temporal_attention": True,
    "use_focal_loss": False,
})

# Full MS-ATCN:
# keep the base configuration unchanged.
write_variant("full_ms_atcn", {})
PY

run_one () {
  local dataset="$1"
  local result_root="$2"
  local variant="$3"

  echo "================================================================================"
  echo "Running ablation: dataset=${dataset}, variant=${variant}"
  echo "================================================================================"

  python -m src.train \
    --config "${ABLATION_CONFIG_DIR}/${variant}.yaml" \
    --dataset "${dataset}" \
    --task_mode binary \
    --data_root datasets \
    --output_dir "${result_root}/${variant}"
}

# CIC-IDS2017
run_one cic_ids2017 results/cic_binary/ablation no_multiscale
run_one cic_ids2017 results/cic_binary/ablation no_attention
run_one cic_ids2017 results/cic_binary/ablation no_focal_loss
run_one cic_ids2017 results/cic_binary/ablation full_ms_atcn

# NSL-KDD
run_one nsl_kdd results/nsl_binary/ablation no_multiscale
run_one nsl_kdd results/nsl_binary/ablation no_attention
run_one nsl_kdd results/nsl_binary/ablation no_focal_loss
run_one nsl_kdd results/nsl_binary/ablation full_ms_atcn

# UNSW-NB15
run_one unsw_nb15 results/unsw_binary_fix/ablation no_multiscale
run_one unsw_nb15 results/unsw_binary_fix/ablation no_attention
run_one unsw_nb15 results/unsw_binary_fix/ablation no_focal_loss
run_one unsw_nb15 results/unsw_binary_fix/ablation full_ms_atcn

echo "================================================================================"
echo "All ablation experiments finished."
echo "================================================================================"
