from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from padufes20.data.schema import PADUFESColumns


def _find_image_path(data_root: Path, img_id: str) -> Optional[Path]:
    candidates = []
    for base in [data_root / "images", data_root]:
        candidates += [
            base / f"{img_id}.png",
            base / f"{img_id}.PNG",
            base / f"{img_id}.jpg",
            base / f"{img_id}.JPG",
        ]
    for p in candidates:
        if p.exists():
            return p
    return None


@dataclass(frozen=True)
class PADUFES20Meta:
    df: pd.DataFrame
    img_col: str
    label_col: str
    patient_col: Optional[str]
    lesion_col: Optional[str]
    classes: List[str]


class PADUFES20Dataset(Dataset):
    def __init__(
        self,
        meta: PADUFES20Meta,
        indices: List[int],
        data_root: str | Path,
        image_size: int = 224,
        augment: bool = False,
    ):
        self.meta = meta
        self.indices = indices
        self.data_root = Path(data_root)

        normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        base = [T.Resize((image_size, image_size)), T.ToTensor(), normalize]

        if augment:
            aug = [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.2),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            ]
            self.transform = T.Compose(aug + base)
        else:
            self.transform = T.Compose(base)

        self.class_to_idx = {c: i for i, c in enumerate(self.meta.classes)}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        row = self.meta.df.iloc[self.indices[i]]
        img_id = str(row[self.meta.img_col])
        img_path = _find_image_path(self.data_root, img_id)
        if img_path is None:
            raise FileNotFoundError(
                f"Could not find image for img_id={img_id}. "
                f"Expected under {self.data_root}/images or {self.data_root}"
            )
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)

        label_str = str(row[self.meta.label_col])
        if label_str not in self.class_to_idx:
            raise ValueError(f"Label '{label_str}' not in {self.meta.classes}")
        y = torch.tensor(self.class_to_idx[label_str], dtype=torch.long)

        sample: Dict[str, torch.Tensor] = {"image": x, "label": y}

        if self.meta.patient_col is not None:
            sample["patient_id"] = torch.tensor(int(row[self.meta.patient_col]), dtype=torch.long)

        return sample


def load_metadata(data_root: str | Path) -> PADUFES20Meta:
    data_root = Path(data_root)
    meta_path = data_root / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at: {meta_path}")

    df = pd.read_csv(meta_path)

    cols = PADUFESColumns()
    img_col = cols.pick(df.columns, cols.img_id_candidates)
    label_col = cols.pick(df.columns, cols.label_candidates)
    patient_col = cols.pick(df.columns, cols.patient_id_candidates)
    lesion_col = cols.pick(df.columns, cols.lesion_id_candidates)

    if img_col is None:
        raise ValueError(f"Could not infer image id column from: {list(df.columns)}")
    if label_col is None:
        raise ValueError(f"Could not infer label column from: {list(df.columns)}")

    classes = sorted(df[label_col].astype(str).unique().tolist())

    return PADUFES20Meta(
        df=df,
        img_col=img_col,
        label_col=label_col,
        patient_col=patient_col,
        lesion_col=lesion_col,
        classes=classes,
    )
