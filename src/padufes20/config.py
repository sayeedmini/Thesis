from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class Config:
    raw: Dict[str, Any]

    @staticmethod
    def load(path: str | Path) -> "Config":
        p = Path(path)
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Config must be a dict, got: {type(raw)}")
        return Config(raw=raw)

    def get(self, key: str, default: Any = None) -> Any:
        parts = key.split(".")
        cur: Any = self.raw
        for part in parts:
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.raw, sort_keys=False)

    def save_resolved(self, out_path: str | Path) -> None:
        Path(out_path).write_text(self.to_yaml(), encoding="utf-8")
