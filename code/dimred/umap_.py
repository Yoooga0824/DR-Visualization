from __future__ import annotations

import numpy as np


def umap_reduce(
    x: np.ndarray,
    *,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """UMAP（umap-learn）。

    依赖：umap-learn
    """
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        raise ValueError("UMAP 输入必须是二维 numpy 数组 [N, D]")

    try:
        import umap
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "缺少依赖 umap-learn，无法运行 UMAP。请安装：pip install umap-learn\n"
            f"原始错误：{e}"
        )

    reducer = umap.UMAP(
        n_components=int(n_components),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=metric,
        random_state=int(random_state),
    )
    z = reducer.fit_transform(x)
    return z.astype(np.float32)
