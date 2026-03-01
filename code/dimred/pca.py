from __future__ import annotations

import numpy as np


def pca_reduce(x: np.ndarray, *, n_components: int = 2, center: bool = True, **kwargs) -> np.ndarray:
    """纯 NumPy PCA（SVD），尽量减少依赖，保证至少一种降维方法可跑通。

    参数：
    - x: [N, D]
    - n_components: 低维维度
    - center: 是否减均值

    返回：
    - z: [N, n_components]
    """
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        raise ValueError("PCA 输入必须是二维 numpy 数组 [N, D]")
    n, d = x.shape
    if n_components <= 0 or n_components > min(n, d):
        raise ValueError(f"n_components={n_components} 非法，应在 [1, min(N,D)] 内")

    x = x.astype(np.float64, copy=False)
    if center:
        mean = np.mean(x, axis=0)
        x0 = x - mean
    else:
        x0 = x

    # SVD: x0 = U S V^T, 主成分方向在 V
    # 使用 full_matrices=False 以减少计算
    u, s, vt = np.linalg.svd(x0, full_matrices=False)
    # 投影到前 k 个主成分
    z = u[:, :n_components] * s[:n_components]
    return z.astype(np.float32)
