from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BoundaryResult:
    boundary_indices: np.ndarray  # shape=(nb,), int64
    normal_indices: np.ndarray  # shape=(nn,), int64
    scores: Optional[np.ndarray] = None  # shape=(n,), float32/float64; higher means more boundary-like
