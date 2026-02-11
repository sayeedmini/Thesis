from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    device: str = "cuda"
    amp: bool = True
    early_stop_patience: int = 5


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}


@torch.no_grad()
def _evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = _to_device(batch, device)
        logits = model(batch["image"])
        loss = criterion(logits, batch["label"])
        total += float(loss.item()) * batch["label"].size(0)
        n += int(batch["label"].size(0))
    return total / max(1, n)


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    ckpt_dir: Path,
) -> Tuple[Path, Dict[str, float]]:
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    best_val = float("inf")
    best_path = ckpt_dir / "best.pt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    patience = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        for batch in pbar:
            batch = _to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                logits = model(batch["image"])
                loss = criterion(logits, batch["label"])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=float(loss.item()))

        scheduler.step()

        val_loss = _evaluate_loss(model, val_loader, criterion, device)
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save({"model_state": model.state_dict(), "val_loss": best_val}, best_path)
        else:
            patience += 1

        if patience >= cfg.early_stop_patience:
            break

    return best_path, {"best_val_loss": best_val}
