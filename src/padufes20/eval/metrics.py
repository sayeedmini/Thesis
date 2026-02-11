from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    log_loss,
)


@dataclass(frozen=True)
class EvalArtifacts:
    scalar: Dict[str, Any]
    per_class: pd.DataFrame
    confusion: np.ndarray


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str]) -> EvalArtifacts:
    if y_prob.ndim != 2:
        raise ValueError(f"y_prob must be 2D, got shape={y_prob.shape}")
    n, c = y_prob.shape
    if c != len(class_names):
        raise ValueError(f"y_prob has C={c} but class_names has {len(class_names)}")

    y_pred = y_prob.argmax(axis=1)

    y_true_bin = np.zeros((n, c), dtype=int)
    y_true_bin[np.arange(n), y_true] = 1

    scalar: Dict[str, Any] = {}
    scalar["accuracy"] = float(accuracy_score(y_true, y_pred))
    scalar["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    scalar["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    scalar["micro_f1"] = float(f1_score(y_true, y_pred, average="micro"))
    scalar["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted"))
    scalar["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    scalar["kappa"] = float(cohen_kappa_score(y_true, y_pred))
    scalar["log_loss"] = float(log_loss(y_true, y_prob, labels=list(range(c))))
    try:
        scalar["macro_auroc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        scalar["macro_auroc_ovr"] = float("nan")
    try:
        scalar["macro_ap"] = float(average_precision_score(y_true_bin, y_prob, average="macro"))
    except Exception:
        scalar["macro_ap"] = float("nan")

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    rows = []
    for i, name in enumerate(class_names):
        auroc = float("nan")
        if len(np.unique(y_true_bin[:, i])) > 1:
            auroc = float(roc_auc_score(y_true_bin[:, i], y_prob[:, i]))
        rows.append(
            {
                "class": name,
                "support": int(report[name]["support"]),
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "ap": float(average_precision_score(y_true_bin[:, i], y_prob[:, i])),
                "auroc": auroc,
            }
        )

    per_class = pd.DataFrame(rows).sort_values("class").reset_index(drop=True)
    conf = confusion_matrix(y_true, y_pred, labels=list(range(c)))
    return EvalArtifacts(scalar=scalar, per_class=per_class, confusion=conf)
