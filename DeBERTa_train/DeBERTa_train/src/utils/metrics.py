
"""Metric helpers for multi-target regression."""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


def compute_regression_metrics(
    predictions: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    *,
    label_names: Iterable[str],
    weights: Mapping[str, float],
) -> Dict[str, float]:
    """Compute MAE/RMSE per label plus weighted aggregates."""
    label_names = list(label_names)
    preds = np.asarray(predictions, dtype=np.float32)
    refs = np.asarray(targets, dtype=np.float32)
    if preds.shape != refs.shape:
        raise ValueError(
            f"Predictions shape {preds.shape} does not match targets shape {refs.shape}"
        )
    metrics: Dict[str, float] = {}
    abs_err = np.abs(preds - refs)
    sq_err = np.square(preds - refs)
    weighted_mae = 0.0
    weighted_rmse = 0.0
    for idx, name in enumerate(label_names):
        mae = float(abs_err[:, idx].mean())
        rmse = float(np.sqrt(sq_err[:, idx].mean()))
        metrics[f"{name}_mae"] = mae
        metrics[f"{name}_rmse"] = rmse
        weight = float(weights.get(name, 0.0))
        weighted_mae += weight * mae
        weighted_rmse += weight * rmse
    metrics["overall_mae"] = weighted_mae
    metrics["overall_rmse"] = weighted_rmse
    return metrics
