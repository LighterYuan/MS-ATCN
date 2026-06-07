# -*- coding: utf-8 -*-
"""
Utilities for saving prediction probabilities using the SAME data pipeline as baselines.

This file should be imported or copied into existing evaluation scripts.
It does not define a new preprocessing pipeline.
It only saves model outputs from the existing validation/test loaders.

Required output columns:
    y_true, y_prob, y_pred

Label convention:
    0 = benign
    1 = attack
"""

from pathlib import Path
import numpy as np
import pandas as pd


def save_binary_predictions(y_true, y_prob_attack, out_path, threshold=0.5):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true).astype(int)
    y_prob_attack = np.asarray(y_prob_attack).astype(float)

    if y_prob_attack.min() < 0 or y_prob_attack.max() > 1:
        raise ValueError("y_prob_attack must be in [0, 1].")

    y_pred = (y_prob_attack >= threshold).astype(int)

    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_prob": y_prob_attack,
            "y_pred": y_pred,
        }
    )
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved binary predictions: {out_path}")


def collect_pytorch_probabilities(model, data_loader, device, output_type="auto"):
    import torch

    model.eval()
    y_true_all = []
    y_prob_all = []

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                raise ValueError("Expected DataLoader batch to be tuple/list: (x, y).")

            x = x.to(device)
            logits = model(x)

            if output_type == "one_logit":
                prob = torch.sigmoid(logits.view(-1))
            elif output_type == "two_logits":
                prob = torch.softmax(logits, dim=1)[:, 1]
            else:
                if logits.ndim == 1 or logits.shape[-1] == 1:
                    prob = torch.sigmoid(logits.view(-1))
                elif logits.shape[-1] == 2:
                    prob = torch.softmax(logits, dim=1)[:, 1]
                else:
                    raise ValueError("Cannot infer output type.")

            y_prob_all.extend(prob.detach().cpu().numpy())
            y_true_all.extend(y.detach().cpu().numpy())

    return np.asarray(y_true_all).astype(int), np.asarray(y_prob_all).astype(float)


def collect_sklearn_probabilities(model, x, y):
    if not hasattr(model, "predict_proba"):
        raise ValueError("The provided model must support predict_proba.")

    prob = model.predict_proba(x)[:, 1]
    return np.asarray(y).astype(int), np.asarray(prob).astype(float)
