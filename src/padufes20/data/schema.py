from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class PADUFESColumns:
    img_id_candidates: List[str] = (
        "img_id",
        "image_id",
        "image",
        "img",
        "imgId",
        "imgID",
    )
    label_candidates: List[str] = (
        "diagnostic",
        "diagnosis",
        "dx",
        "label",
        "target",
        "lesion_type",
    )
    patient_id_candidates: List[str] = (
        "patient_id",
        "patient",
        "patientId",
        "patientID",
        "id_patient",
    )
    lesion_id_candidates: List[str] = (
        "lesion_id",
        "lesion",
        "lesionId",
        "lesionID",
        "id_lesion",
    )

    def pick(self, available: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
        avail = {c.lower(): c for c in available}
        for cand in candidates:
            key = cand.lower()
            if key in avail:
                return avail[key]
        return None
