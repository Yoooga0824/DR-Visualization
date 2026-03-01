from __future__ import annotations

import numpy as np

from .types import BoundaryResult


def detect_bdlle(
    x: np.ndarray,
    *,
    d: int,
    k: int,
    threshold_mode: str = "ratio",
    threshold_ratio: float = 0.7,
    **kwargs,
) -> BoundaryResult:
    """BD-LLE 边界检测包装器。

    - scores 取自 BDLLE 算法输出的 B 指示器（越大越边界）。
    - threshold_mode:
      - internal: 使用 BDLLE.py 内部 threshold（max/2）得到的 boundary_bool
      - ratio: 使用 ratio*max(scores)
    """
    from .BDLLE import bd_lle

    boundary_bool, scores = bd_lle(x, d=int(d), K=int(k))

    threshold_mode = (threshold_mode or "internal").lower()
    if threshold_mode == "internal":
        boundary_mask = boundary_bool.astype(bool)
    elif threshold_mode == "ratio":
        max_b = float(np.max(scores)) if scores.size else 0.0
        thr = float(threshold_ratio) * max_b
        boundary_mask = scores >= thr
    else:
        raise ValueError("threshold_mode 仅支持 internal 或 ratio")

    boundary_indices = np.where(boundary_mask)[0].astype(np.int64)
    normal_indices = np.where(~boundary_mask)[0].astype(np.int64)
    return BoundaryResult(boundary_indices=boundary_indices, normal_indices=normal_indices, scores=scores)
