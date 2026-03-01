from __future__ import annotations

import numpy as np

from .types import BoundaryResult


def detect_ocsvm(
    x: np.ndarray,
    *,
    nu: float = 0.05,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    threshold_mode: str = "internal",
    threshold_ratio: float = 0.7,
    random_state: int = 42,
    **kwargs,
) -> BoundaryResult:
    """One-Class SVM 边界检测。

    直觉：OCSVM 学到数据支持集边界；被判为 -1 的点可视为边界/外侧。

    scores：使用 -decision_function，使得 score 越大越“边界/外侧”。

    threshold_mode:
    - internal: 使用 OCSVM predict==-1
    - ratio: 使用 ratio*max(score)

    依赖：scikit-learn
    """
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        raise ValueError("输入必须是二维 numpy 数组 [N, D]")

    try:
        from sklearn.svm import OneClassSVM
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "缺少依赖 scikit-learn，无法运行 ocsvm。请安装：pip install scikit-learn\n"
            f"原始错误：{e}"
        )

    model = OneClassSVM(nu=float(nu), kernel=str(kernel), gamma=gamma)
    model.fit(x)

    # decision_function: 越大越在内；我们取负号作为“边界/外侧分数”
    decision = model.decision_function(x).astype(np.float64, copy=False)
    scores = -decision

    threshold_mode = (threshold_mode or "internal").lower()
    if threshold_mode == "internal":
        pred = model.predict(x)
        boundary_mask = pred == -1
    elif threshold_mode == "ratio":
        max_s = float(np.max(scores)) if scores.size else 0.0
        thr = float(threshold_ratio) * max_s
        boundary_mask = scores >= thr
    else:
        raise ValueError("threshold_mode 仅支持 internal 或 ratio")

    boundary_indices = np.where(boundary_mask)[0].astype(np.int64)
    normal_indices = np.where(~boundary_mask)[0].astype(np.int64)
    return BoundaryResult(boundary_indices=boundary_indices, normal_indices=normal_indices, scores=scores)
