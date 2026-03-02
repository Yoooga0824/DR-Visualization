import numpy as np
import sys
import os
import time
import json
import argparse
from typing import Optional, Tuple

from html_report import generate_interactive_html

# 参数化前缀

parser = argparse.ArgumentParser(description='交互式边界分析可视化，支持多降维方法和多数据集')
parser.add_argument('--prefix', type=str, default='TSNE', help='降维方法前缀，如 TSNE、NeuralTSNE、UMAP')
parser.add_argument('--detector', type=str, default='bdlle', help='边界检测方法名（用于读取已保存的边界结果文件，如 bdlle/knn_distance/ocsvm）')
parser.add_argument('--features-dir', type=str, default=None, help='features 目录（如 data/Hands-features），自动推断数据集名')
parser.add_argument('--results-dir', type=str, default=None, help='结果保存目录（如 results/Hands-results），如未指定自动推断')
parser.add_argument('--img-base', type=str, default=None, help='图片服务基地址，如 http://172.16.57.85:5678（file:// 环境下请不要使用 window.location）')
args = parser.parse_args()
prefix = args.prefix
detector = (args.detector or 'bdlle').strip()
# 默认本地服务器，避免生成 HTML 指向内网 IP
image_api_base = args.img_base if args.img_base else 'http://localhost:5678'

# 自动推断 features 目录：优先参数，其次自动在 data 下寻找唯一 *-features 目录
if args.features_dir:
    features_dir = os.path.abspath(args.features_dir)
else:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_root = os.path.join(project_root, 'data')
    candidates = []
    if os.path.isdir(data_root):
        for name in os.listdir(data_root):
            p = os.path.join(data_root, name)
            if os.path.isdir(p) and name.endswith('-features'):
                candidates.append(p)
    if len(candidates) == 1:
        features_dir = os.path.abspath(candidates[0])
        print(f"未指定 --features-dir，自动选择: {features_dir}")
    elif len(candidates) > 1:
        raise RuntimeError(f"检测到多个 *-features 目录: {candidates}，请使用 --features-dir 指定其一。")
    else:
        raise RuntimeError(f"未在 {data_root} 下找到任何 *-features 目录，请检查目录结构或传入 --features-dir。")

# 自动推断数据集名（如 Hands-features -> Hands, yalefaces-features -> yalefaces）
dataset_name = os.path.basename(features_dir).replace('-features', '')

# 结果保存目录
if args.results_dir:
    RESULTS_DIR = args.results_dir
else:
    results_root = os.path.abspath(os.path.join(features_dir, '..', '..', 'results'))
    RESULTS_DIR = os.path.join(results_root, f'{dataset_name}-results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 兼容：旧版页面顶部曾展示 K/Threshold。
# 新版页面读取 detect_boundaries.py 产生的边界结果文件，因此这里不再假定固定 K/Threshold。

# 添加自定义模块路径（如果需要可以保留）
sys.path.append(os.path.abspath('./code/'))

def _try_import_wasserstein_loss():
    """优先使用仓库已有的 wasserstein_loss（geomloss），失败则返回 None。"""
    try:
        from tool_functions import wasserstein_loss  # type: ignore

        return wasserstein_loss
    except Exception:
        return None


def _wasserstein_fallback(a: np.ndarray, b: np.ndarray) -> float:
    """无 torch/geomloss 时的轻量 fallback：按维度计算 1D Wasserstein 并取均值。"""
    try:
        from scipy.stats import wasserstein_distance
    except Exception as e:
        raise RuntimeError(f"缺少 scipy.stats.wasserstein_distance，无法计算 Wasserstein 距离：{e}")

    if a.size == 0 or b.size == 0:
        return float('nan')

    dims = min(a.shape[1], b.shape[1])
    vals = [float(wasserstein_distance(a[:, i], b[:, i])) for i in range(dims)]
    return float(np.mean(vals))


def compute_wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个点集之间的 Wasserstein 距离（优先 geomloss，否则 fallback）。"""
    wloss = _try_import_wasserstein_loss()
    if wloss is None:
        return _wasserstein_fallback(a, b)

    # geomloss 版本需要 torch tensor
    try:
        import torch

        ta = torch.tensor(a, dtype=torch.float32)
        tb = torch.tensor(b, dtype=torch.float32)
        val = wloss(ta, tb)
        return float(val.detach().cpu().item())
    except Exception:
        # 如果 torch/geomloss 任何一环失败，回退到 scipy 的按维度 1D Wasserstein
        return _wasserstein_fallback(a, b)


def _load_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"无法读取为 numpy.ndarray: {path}")
    if arr.ndim != 2:
        raise ValueError(f"期望二维数组 [N, D]，实际 {arr.ndim} 维：{path}")
    return arr


def _find_features_file() -> str:
    """优先 features.npy，兼容 <dataset>_features.npy。"""
    p = os.path.join(features_dir, 'features.npy')
    if os.path.exists(p):
        return p
    legacy = os.path.join(features_dir, f'{dataset_name}_features.npy')
    if os.path.exists(legacy):
        return legacy
    raise FileNotFoundError(f"未找到高维特征文件：{p}（或兼容旧命名 {legacy}）")


def _row_view(arr: np.ndarray) -> np.ndarray:
    """将二维数组按行视为定长 bytes，便于做 isin。"""
    a = np.ascontiguousarray(arr)
    if a.ndim != 2:
        raise ValueError("仅支持二维数组")
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def indices_from_subset(full: np.ndarray, subset: np.ndarray) -> np.ndarray:
    """给定 full（[N,D]）与 subset（[M,D]，来自 full 的行子集），返回 subset 对应在 full 中的索引集合。"""
    if subset.size == 0:
        return np.array([], dtype=np.int64)
    if full.ndim != 2 or subset.ndim != 2 or full.shape[1] != subset.shape[1]:
        raise ValueError("full/subset 维度不匹配")
    full_v = _row_view(full)
    subset_v = _row_view(subset)
    mask = np.isin(full_v, subset_v)
    return np.where(mask)[0].astype(np.int64)



def load_and_normalize_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载低维坐标并归一化到 [0,1]，同时返回 labels（用于颜色：normal/anomalous）。

    返回：(raw_embedding, normalized_embedding, labels)
    """
    print("加载并归一化低维数据...")

    # 低维坐标（优先 *_features.npy，兼容旧 *_embedding.npy）
    normal_path = os.path.join(features_dir, f'{prefix}_features.npy')
    if not os.path.exists(normal_path):
        normal_path = os.path.join(features_dir, f'{prefix}_embedding.npy')
    raw_embedding = _load_npy(normal_path)

    # 兼容旧数据：如果存在 anomalous 前缀则用于区分颜色；否则全视为 normal
    anom_path = os.path.join(features_dir, f'anomalous_{prefix}_features.npy')
    if not os.path.exists(anom_path):
        anom_path = os.path.join(features_dir, f'anomalous_{prefix}_embedding.npy')

    anomalous_embedding = None
    if os.path.exists(anom_path):
        anomalous_embedding = _load_npy(anom_path)

    if anomalous_embedding is not None:
        combined = np.vstack([raw_embedding, anomalous_embedding])
        labels = np.concatenate([np.zeros(len(raw_embedding)), np.ones(len(anomalous_embedding))])
    else:
        combined = raw_embedding
        labels = np.zeros(len(raw_embedding))

    # 归一化到[0,1]范围（用于可视化/距离度量的尺度统一）
    min_vals = np.min(combined, axis=0)
    max_vals = np.max(combined, axis=0)
    denom = (max_vals - min_vals)
    denom[denom == 0] = 1.0
    normalized = (combined - min_vals) / denom

    print(f"低维坐标形状: normal={raw_embedding.shape}（combined={combined.shape}）")
    return combined, normalized, labels

def load_boundary_files() -> Tuple[np.ndarray, np.ndarray]:
    """读取已保存的边界点子集（高维/低维）。"""
    det = detector.lower()
    high_boundary_path = os.path.join(features_dir, f'features_{det}_boundary.npy')
    low_boundary_path = os.path.join(features_dir, f'{prefix}_{det}_boundary.npy')

    if not os.path.exists(high_boundary_path):
        raise FileNotFoundError(
            f"未找到原空间边界文件：{high_boundary_path}\n"
            "请先运行 detect_boundaries.py 生成 features_<detector>_boundary.npy"
        )
    if not os.path.exists(low_boundary_path):
        raise FileNotFoundError(
            f"未找到隐空间边界文件：{low_boundary_path}\n"
            "请先运行 detect_boundaries.py 生成 <method>_<detector>_boundary.npy"
        )

    high_boundary = _load_npy(high_boundary_path)
    low_boundary = _load_npy(low_boundary_path)
    return high_boundary, low_boundary

def calculate_wasserstein_metrics(
    *,
    normalized_low: np.ndarray,
    high_boundary_indices: np.ndarray,
    low_boundary_indices: np.ndarray,
) -> float:
    """将“原空间边界点”和“隐空间边界点”都投到隐空间坐标上，计算二者的 Wasserstein 距离。"""
    if high_boundary_indices.size == 0 or low_boundary_indices.size == 0:
        return float('nan')
    a = normalized_low[high_boundary_indices]
    b = normalized_low[low_boundary_indices]
    return compute_wasserstein(a, b)



def main():
    print("=" * 50)
    print(f"边界分析（读取已保存边界结果）：prefix={prefix}, detector={detector}")
    print("=" * 50)

    total_start_time = time.time()

    low_raw, low_norm, labels = load_and_normalize_data()

    # 读取 mapping，严格对齐长度（避免映射/绘图/指标错位）
    mapping_path = os.path.join(RESULTS_DIR, f'embedding_to_image_mapping_{prefix}.json')
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    N = len(mapping_data)
    low_raw = low_raw[:N]
    low_norm = low_norm[:N]
    labels = labels[:N]

    # 读取边界点子集文件
    high_boundary_points, low_boundary_points = load_boundary_files()

    # 1) 隐空间边界：用低维边界点子集反推索引，用于散点图高亮
    try:
        low_boundary_indices = indices_from_subset(low_raw, low_boundary_points)
    except Exception:
        low_boundary_indices = np.array([], dtype=np.int64)

    # 2) 原空间边界：用高维边界点子集反推索引 -> 再映射到低维坐标上用于 WD
    try:
        high_features_path = _find_features_file()
        high_full = np.load(high_features_path, mmap_mode='r')
        high_boundary_indices = indices_from_subset(np.asarray(high_full), high_boundary_points)
    except Exception:
        high_boundary_indices = np.array([], dtype=np.int64)

    wdist = calculate_wasserstein_metrics(
        normalized_low=low_norm,
        high_boundary_indices=high_boundary_indices,
        low_boundary_indices=low_boundary_indices,
    )

    metrics = {
        'total_count': int(N),
        'orig_boundary_count': int(high_boundary_points.shape[0]),
        'latent_boundary_count': int(low_boundary_points.shape[0]),
        'wasserstein': float(wdist),
    }

    html_path = generate_interactive_html(
        low_norm,
        high_boundary_indices=high_boundary_indices,
        low_boundary_indices=low_boundary_indices,
        metrics=metrics,
        results_dir=RESULTS_DIR,
        prefix=prefix,
        detector=detector,
        image_api_base=image_api_base,
        dataset_name=dataset_name,
    )

    total_time = time.time() - total_start_time
    print("=" * 50)
    print("分析完成!")
    print(f"总耗时: {total_time:.2f}秒 ({total_time / 60:.1f}分钟)")
    print(f"总样本数: {metrics['total_count']}")
    print(f"原空间边界点数: {metrics['orig_boundary_count']}")
    print(f"隐空间边界点数: {metrics['latent_boundary_count']}")
    print(f"Wasserstein距离: {metrics['wasserstein']:.6f}")
    print(f"交互式HTML保存至: {html_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()