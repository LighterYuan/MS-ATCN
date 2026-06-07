#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

CONFIG="configs/base.yaml"
DATA_ROOT="datasets"

for DATASET in cic_ids2017 nsl_kdd unsw_nb15; do
  if [ "$DATASET" = "cic_ids2017" ]; then
    OUT_BASE="results/cic_binary"
  elif [ "$DATASET" = "nsl_kdd" ]; then
    OUT_BASE="results/nsl_binary"
  else
    OUT_BASE="results/unsw_binary_fix"
  fi

  for MODEL in mlp cnn lstm gru tcn xgboost lightgbm; do
    echo "Running ${MODEL} on ${DATASET}"
    python -m src.train_baseline \
      --config "$CONFIG" \
      --dataset "$DATASET" \
      --task_mode binary \
      --data_root "$DATA_ROOT" \
      --model "$MODEL" \
      --output_dir "${OUT_BASE}/${MODEL}"
  done
done
