# MS-ATCN Decision Refinement Code

This is the revised supplement code for the paper.  
It is designed to serve the proposed model framework, not to modify the model only for higher scores.

## Method name

Recommended paper name:

```text
MS-ATCN-DR: MS-ATCN with Validation-based Tree Decision Refinement
```

or:

```text
MS-ATCN-LightGBM-DR
MS-ATCN-XGBoost-DR
```

## Design principle

MS-ATCN remains the core proposed model.  
The tree model is used as a decision refinement module because tabular flow-level IDS features often benefit from tree-based decision boundaries.

The refinement is performed only at the posterior probability level:

```text
p_refined = alpha * p_ms_atcn + (1 - alpha) * p_tree
```

## Strict experimental protocol

1. MS-ATCN, XGBoost, and LightGBM must use the same preprocessing pipeline.
2. They must use the same train/validation/test split.
3. UNSW-NB15 must remove `attack_cat` and `id` before model training and prediction generation.
4. Alpha must be selected only on the validation set.
5. Test set is evaluated once using the selected alpha.
6. Outputs remain consistent with your existing experiments:
   - `metrics.json`
   - `classification_report.txt`
   - `confusion_matrix.csv`

## Files

- `run_decision_refinement.py`: main script.
- `save_prediction_outputs.py`: helper functions for saving probabilities.
- `integration_example.py`: integration example for existing code.
- `README_DECISION_REFINEMENT.md`: this file.

## Example: UNSW-NB15 + LightGBM decision refinement

```bash
python run_decision_refinement.py   --dataset unsw_binary_fix   --tree-name LightGBM   --ms-val results/unsw_binary_fix/MS-ATCN/val_predictions.csv   --tree-val results/unsw_binary_fix/LightGBM/val_predictions.csv   --ms-test results/unsw_binary_fix/MS-ATCN/test_predictions.csv   --tree-test results/unsw_binary_fix/LightGBM/test_predictions.csv   --out-dir results/unsw_binary_fix/ms_atcn_lightgbm_dr   --select-metric Macro-F1
```

## Example: UNSW-NB15 + XGBoost decision refinement

```bash
python run_decision_refinement.py   --dataset unsw_binary_fix   --tree-name XGBoost   --ms-val results/unsw_binary_fix/MS-ATCN/val_predictions.csv   --tree-val results/unsw_binary_fix/XGBoost/val_predictions.csv   --ms-test results/unsw_binary_fix/MS-ATCN/test_predictions.csv   --tree-test results/unsw_binary_fix/XGBoost/test_predictions.csv   --out-dir results/unsw_binary_fix/ms_atcn_xgboost_dr   --select-metric Macro-F1
```

## Suggested paper wording

> To better align the neural representation ability of MS-ATCN with the decision characteristics of tabular flow-level IDS features, a validation-based decision refinement module is introduced. The module combines the posterior attack probability of MS-ATCN with that of a tree-based classifier using a scalar coefficient selected only on the validation set. No additional feature engineering, data resampling, or test-set tuning is introduced.

## Important note

If the selected alpha is 0.0, the validation protocol indicates that the tree model alone is better under the selected metric. Do not hide this result.
