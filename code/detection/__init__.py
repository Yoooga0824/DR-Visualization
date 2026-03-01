"""边界/检测方法包（Boundary Detectors）。

约定：
- 检测器输入 X: np.ndarray, shape=(N, D)
- 输出 BoundaryResult：
  - boundary_indices: 1D int
  - normal_indices: 1D int
  - scores: (可选) 每个点的连续分数，数值越大越“边界”

统一入口：
- list_detectors()
- get_detector(name)

现有算法：
- BDLLE: 见 BDLLE.py（算法本体）
- bdlle / knn_distance / ocsvm: 见 registry.py 注册的检测器
"""

from .registry import BoundaryResult, get_detector, list_detectors

__all__ = ["BoundaryResult", "get_detector", "list_detectors"]
