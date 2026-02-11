from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    val_size: float = 0.1
    seed: int = 42


def _pick_holdout_via_sgkf(
    y: np.ndarray, groups: np.ndarray, holdout_size: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    n_splits = int(round(1.0 / holdout_size))
    n_splits = max(2, n_splits)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, holdout_idx = next(sgkf.split(X=np.zeros_like(y), y=y, groups=groups))
    return train_idx, holdout_idx


def make_patientwise_splits(
    df: pd.DataFrame,
    label_col: str,
    group_col: Optional[str],
    cfg: SplitConfig,
) -> Dict[str, List[int]]:
    y = df[label_col].astype(str).to_numpy()
    groups = np.arange(len(df)) if group_col is None else df[group_col].to_numpy()

    trainval_idx, test_idx = _pick_holdout_via_sgkf(y, groups, cfg.test_size, cfg.seed)

    df_tv = df.iloc[trainval_idx].reset_index(drop=False)  # original index saved to 'index'
    y_tv = df_tv[label_col].astype(str).to_numpy()
    g_tv = np.arange(len(df_tv)) if group_col is None else df_tv[group_col].to_numpy()

    val_rel = cfg.val_size / (1.0 - cfg.test_size)
    tv_train_idx, val_local = _pick_holdout_via_sgkf(y_tv, g_tv, val_rel, cfg.seed + 1)

    train_idx = df_tv.iloc[tv_train_idx]["index"].to_numpy()
    val_idx = df_tv.iloc[val_local]["index"].to_numpy()

    return {"train": train_idx.astype(int).tolist(), "val": val_idx.astype(int).tolist(), "test": test_idx.astype(int).tolist()}


def assert_no_group_leakage(df: pd.DataFrame, split: Dict[str, List[int]], group_col: Optional[str]) -> None:
    if group_col is None:
        return

    def groups_of(key: str) -> set:
        return set(df.iloc[split[key]][group_col].tolist())

    tr, va, te = groups_of("train"), groups_of("val"), groups_of("test")
    if tr & va:
        raise AssertionError(f"Group leakage train∩val: {len(tr & va)} groups overlap.")
    if tr & te:
        raise AssertionError(f"Group leakage train∩test: {len(tr & te)} groups overlap.")
    if va & te:
        raise AssertionError(f"Group leakage val∩test: {len(va & te)} groups overlap.")
