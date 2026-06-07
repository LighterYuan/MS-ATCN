# Split-Safe Data Loading Revision

This revision modifies `src/data.py` to enforce leakage-aware sliding-window construction.

## Main changes

1. For single-dataframe datasets such as CIC-IDS2017:
   - row-level train/validation/test split is performed first;
   - categorical mappings, medians, and scaler are fitted on the training split only;
   - sliding windows are constructed independently inside train, validation, and test splits;
   - no window can cross a split boundary.

2. For official train/test datasets such as UNSW-NB15 and NSL-KDD:
   - existing public interfaces are preserved;
   - categorical mappings are now fitted on training features only;
   - validation is still derived from the official training partition;
   - windows are constructed independently within train, validation, and test.

3. Compatibility:
   - `prepare_dataloaders()` signature is unchanged;
   - `WindowSequenceDataset` output remains `[num_features, seq_len]`;
   - MS-ATCN training scripts remain compatible;
   - tree baselines remain compatible because `loader_to_numpy()` still flattens window-level samples;
   - output files such as `metrics.json`, `classification_report.txt`, `confusion_matrix.csv`,
     `label_mapping.json`, and `data_info.json` remain supported.

## Important note

Because CIC-IDS2017 now prevents split-boundary window overlap and prevents scaler/imputer fitting
on validation/test data, CIC-IDS2017 metrics may differ from earlier runs. This is expected and is
methodologically preferable for a leakage-aware manuscript.
