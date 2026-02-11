from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    ece: float
    mce: float
    bin_acc: np.ndarray
    bin_conf: np.ndarray
    bin_count: np.ndarray


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> CalibrationResult:
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_acc = np.zeros(n_bins, dtype=float)
    bin_conf = np.zeros(n_bins, dtype=float)
    bin_count = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        bin_count[b] = int(mask.sum())
        bin_acc[b] = float(correct[mask].mean())
        bin_conf[b] = float(conf[mask].mean())

    n = len(y_true)
    gaps = np.abs(bin_acc - bin_conf)
    ece = float((bin_count / max(1, n) * gaps).sum())
    mce = float(gaps.max()) if n > 0 else float("nan")
    return CalibrationResult(ece=ece, mce=mce, bin_acc=bin_acc, bin_conf=bin_conf, bin_count=bin_count)
