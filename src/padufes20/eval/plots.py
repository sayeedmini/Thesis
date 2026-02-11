from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(conf: np.ndarray, class_names: List[str], out: Path, normalize: bool = True) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cm = conf.astype(float)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{int(conf[i, j])}", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_reliability_diagram(bin_conf: np.ndarray, bin_acc: np.ndarray, bin_count: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], linestyle="--")

    sizes = 30 + 170 * (bin_count / max(1, bin_count.max()))
    ax.scatter(bin_conf, bin_acc, s=sizes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram (max-prob multiclass)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
