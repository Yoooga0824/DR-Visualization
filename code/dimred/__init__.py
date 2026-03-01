"""降维（Dimensionality Reduction）模块。

目标：
- 将高维特征矩阵 X [N, D] 降到低维 Z [N, n_components]（通常 2 维用于可视化）。
- 输出命名与仓库约定对齐：<prefix>_features.npy（兼容旧 *_embedding.npy）。

本模块只提供算法函数，不做文件 I/O；文件读写由 code/compute_embeddings.py 负责。
"""

from .registry import get_reducer, list_methods

__all__ = ["get_reducer", "list_methods"]
