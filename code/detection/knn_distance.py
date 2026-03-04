from __future__ import annotations

import numpy as np

from .types import BoundaryResult


# 全局默认 K：不显式传参时将使用该值。
# 你只需要改这里，就能影响 knn_distance 的默认行为。
KNN_DISTANCE_K = 50


def detect_knn_distance(
    x: np.ndarray,
    *,
    k: int | None = None,
    threshold_mode: str = "ratio",
    threshold_ratio: float = 0.7,
    **kwargs,
) -> BoundaryResult:
    """KNN 距离边界检测（通用、快速、依赖少）。

    思路：
    - 用 k 近邻距离的均值作为 score：score 越大，点越处在稀疏/边缘区域。

    threshold_mode:
    - internal: 使用 80% 分位数作为阈值（经验值）
    - ratio: 使用 ratio*max(score) 作为阈值

    参数 k:
    - 若不传（或传 None），默认使用本模块全局变量 KNN_DISTANCE_K。
    """
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        raise ValueError("输入必须是二维 numpy 数组 [N, D]")

    n = x.shape[0]
    if n < 2:
        boundary_indices = np.arange(n, dtype=np.int64)
        normal_indices = np.array([], dtype=np.int64)
        return BoundaryResult(boundary_indices=boundary_indices, normal_indices=normal_indices, scores=np.zeros(n))

    if k is None:
        k = KNN_DISTANCE_K
    k = int(k)
    if k <= 0:
        raise ValueError("k 必须为正")
    k = min(k, n - 1)

    # 优先用 scipy 的 cKDTree（仓库已用 scipy）
    try:
        from scipy.spatial import cKDTree
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"缺少 scipy，无法运行 knn_distance：{e}")

    tree = cKDTree(x)
    # 查 k+1（包含自身），然后丢弃第 0 个
    dists, _ = tree.query(x, k=k + 1)
    dists = dists[:, 1:]
    scores = np.mean(dists, axis=1)

    threshold_mode = (threshold_mode or "ratio").lower()
    if threshold_mode == "internal":
        thr = float(np.quantile(scores, 0.8))
        boundary_mask = scores >= thr
    elif threshold_mode == "ratio":
        max_s = float(np.max(scores)) if scores.size else 0.0
        thr = float(threshold_ratio) * max_s
        boundary_mask = scores >= thr
    else:
        raise ValueError("threshold_mode 仅支持 internal 或 ratio")

    boundary_indices = np.where(boundary_mask)[0].astype(np.int64)
    normal_indices = np.where(~boundary_mask)[0].astype(np.int64)
    return BoundaryResult(boundary_indices=boundary_indices, normal_indices=normal_indices, scores=scores)
