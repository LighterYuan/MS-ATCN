# -*- coding: utf-8 -*-
"""
Example integration file.

This file shows where to add probability saving in existing experiment code.
It is intentionally not a standalone training script because the decision
refinement module must use the same data interface as baselines.
"""

# from save_prediction_outputs import (
#     save_binary_predictions,
#     collect_pytorch_probabilities,
#     collect_sklearn_probabilities,
# )

# MS-ATCN validation/test probability saving:
#
# y_val, p_val = collect_pytorch_probabilities(ms_atcn_model, val_loader, device)
# save_binary_predictions(y_val, p_val, "results/unsw_binary_fix/MS-ATCN/val_predictions.csv")
#
# y_test, p_test = collect_pytorch_probabilities(ms_atcn_model, test_loader, device)
# save_binary_predictions(y_test, p_test, "results/unsw_binary_fix/MS-ATCN/test_predictions.csv")

# LightGBM validation/test probability saving:
#
# y_val, p_val = collect_sklearn_probabilities(lightgbm_model, X_val, y_val)
# save_binary_predictions(y_val, p_val, "results/unsw_binary_fix/LightGBM/val_predictions.csv")
#
# y_test, p_test = collect_sklearn_probabilities(lightgbm_model, X_test, y_test)
# save_binary_predictions(y_test, p_test, "results/unsw_binary_fix/LightGBM/test_predictions.csv")

# XGBoost validation/test probability saving:
#
# y_val, p_val = collect_sklearn_probabilities(xgboost_model, X_val, y_val)
# save_binary_predictions(y_val, p_val, "results/unsw_binary_fix/XGBoost/val_predictions.csv")
#
# y_test, p_test = collect_sklearn_probabilities(xgboost_model, X_test, y_test)
# save_binary_predictions(y_test, p_test, "results/unsw_binary_fix/XGBoost/test_predictions.csv")
