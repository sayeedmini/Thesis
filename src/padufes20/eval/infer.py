from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass(frozen=True)
class InferConfig:
    device: str = "cuda"


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, cfg: InferConfig) -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")
    model.to(device)
    model.eval()

    probs, labels = [], []
    for batch in tqdm(loader, desc="infer", leave=False):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].cpu().numpy()
        logits = model(x)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        labels.append(y)

    return np.concatenate(labels, axis=0), np.concatenate(probs, axis=0)
