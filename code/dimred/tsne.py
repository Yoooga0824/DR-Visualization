from __future__ import annotations

import numpy as np


def tsne_reduce(
    x: np.ndarray,
    *,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: str | float = "auto",
    n_iter: int = 1000,
    random_state: int = 42,
    init: str = "pca",
    **kwargs,
) -> np.ndarray:
    """sklearn TSNE。

    说明：TSNE 计算较慢；建议先用 PCA 验证流程，再用 TSNE/UMAP。
    依赖：scikit-learn
    """
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        raise ValueError("TSNE 输入必须是二维 numpy 数组 [N, D]")

    try:
        from sklearn.manifold import TSNE
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "缺少依赖 scikit-learn，无法运行 TSNE。请安装：pip install scikit-learn\n"
            f"原始错误：{e}"
        )

    model = TSNE(
        n_components=int(n_components),
        perplexity=float(perplexity),
        learning_rate=learning_rate,
        max_iter=int(n_iter),
        init=init,
        random_state=int(random_state),
    )
    z = model.fit_transform(x)
    return z.astype(np.float32)
