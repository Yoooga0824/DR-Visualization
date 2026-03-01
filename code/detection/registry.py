from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from .types import BoundaryResult
from .bdlle_detector import detect_bdlle
from .knn_distance import detect_knn_distance
from .ocsvm import detect_ocsvm


DetectorFn = Callable[..., BoundaryResult]


_DETECTORS: Dict[str, DetectorFn] = {
    "bdlle": detect_bdlle,
    "knn_distance": detect_knn_distance,
    "ocsvm": detect_ocsvm,
}


def list_detectors() -> List[str]:
    return sorted(_DETECTORS.keys())


def get_detector(name: str) -> DetectorFn:
    key = (name or "").strip().lower()
    if key not in _DETECTORS:
        raise KeyError(f"未知检测方法: {name}（可用：{list_detectors()}）")
    return _DETECTORS[key]
