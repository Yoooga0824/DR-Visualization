"""detect_boundaries.py

通用边界检测脚本：对高维特征（features.npy）与/或低维特征（<方法>_features.npy）执行边界检测，
并按你的命名规范保存“点坐标子集”。

输出文件（默认保存到高维文件所在目录，即 data/<dataset>-features）：
- 高维：
    - features_<检测方法>_boundary.npy
    - features_<检测方法>_normal.npy
- 低维：
    - <降维方法>_<检测方法>_boundary.npy
    - <降维方法>_<检测方法>_normal.npy

当前内置检测器：
- bdlle：基于 LLE 的边界检测（见 detection/BDLLE.py）

示例：
    # 只在高维 features.npy 上检测边界
    py code/detect_boundaries.py --high data/Hands-features/features.npy --detector bdlle --k 100 --d 2

    # 同时在高维与低维上检测边界（低维通常是 2D）
  py code/detect_boundaries.py \
        --high data/Hands-features/features.npy \
        --low  data/Hands-features/TSNE_features.npy \
    --detector bdlle --k-high 150 --k-low 100 --d 2 --threshold-mode ratio --threshold-ratio 0.7

注意：
- 本脚本不做降维，只读取你已生成的 *_features.npy（或兼容旧的 *_embedding.npy）。
- 检测器通过 detection/registry.py 统一入口扩展。
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


from detection import BoundaryResult, get_detector, list_detectors


def natural_key(name: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def load_array(path: str) -> np.ndarray:
    arr = np.load(path)
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"无法读取为 numpy.ndarray: {path}")
    if arr.ndim != 2:
        raise ValueError(f"期望二维数组 [N, D]，实际 {arr.ndim} 维：{path}")
    return arr


def normalize_array(x: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "none").lower()
    if mode == "none":
        return x
    if mode == "minmax":
        min_vals = np.min(x, axis=0)
        max_vals = np.max(x, axis=0)
        denom = (max_vals - min_vals)
        denom[denom == 0] = 1.0
        return (x - min_vals) / denom
    if mode == "zscore":
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        std[std == 0] = 1.0
        return (x - mean) / std
    raise ValueError(f"未知 normalize 模式: {mode}（支持：none|minmax|zscore）")


"""detect_boundaries.py 使用 detection/registry.py 中的统一检测器入口。"""


def infer_method_from_path(path: str) -> str:
    """从文件名推断“方法名”。

    - TSNE_features.npy  -> TSNE
    - TSNE_embedding.npy -> TSNE（兼容旧命名）
    - features.npy       -> features
    """
    name = Path(path).stem
    if name.endswith("_features"):
        name = name[: -len("_features")]
    if name.endswith("_embedding"):
        name = name[: -len("_embedding")]
    return name


def save_points(out_dir: Path, *, tag: str, detector_name: str, x: np.ndarray, result: BoundaryResult) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    boundary = x[result.boundary_indices]
    normal = x[result.normal_indices]
    boundary_path = out_dir / f"{tag}_{detector_name}_boundary.npy"
    normal_path = out_dir / f"{tag}_{detector_name}_normal.npy"
    # 重要：保持与原输入一致的 dtype，便于后续精确回查索引（用于指标计算/对齐验证）。
    np.save(boundary_path, boundary)
    np.save(normal_path, normal)
    return boundary_path, normal_path


def main():
    parser = argparse.ArgumentParser(description="通用边界检测：对高维/低维数据执行边界检测并保存点坐标子集")

    parser.add_argument("--high", type=str, required=True, help="高维特征 .npy（建议 data/<dataset>-features/features.npy，形如 [N, D]）")
    parser.add_argument("--low", type=str, default=None, help="低维特征 .npy（<方法>_features.npy；兼容旧 *_embedding.npy，形如 [N, 2] 或 [N, d]）")

    parser.add_argument("--skip-high", action="store_true", help="跳过高维边界检测与输出（仅在同时传了 --low 时有意义）")
    parser.add_argument("--skip-low", action="store_true", help="跳过低维边界检测与输出")

    parser.add_argument("--detector", type=str, default="bdlle", help=f"边界检测方法（可用：{list_detectors()}）")

    parser.add_argument("--d", type=int, default=2, help="流形维度 d（BDLLE 所需）")
    parser.add_argument("--k", type=int, default=100, help="通用 K（若未分别指定 k-high/k-low，则都使用该值）")
    parser.add_argument("--k-high", type=int, default=None, help="高维检测的 K（覆盖 --k）")
    parser.add_argument("--k-low", type=int, default=None, help="低维检测的 K（覆盖 --k）")

    parser.add_argument(
        "--threshold-mode",
        type=str,
        default="ratio",
        help="internal: 使用 detector 内部默认阈值；ratio: 使用 ratio*max(score)（默认 ratio）",
    )
    parser.add_argument("--threshold-ratio", type=float, default=0.7, help="当 threshold-mode=ratio 时生效")

    parser.add_argument(
        "--normalize-high",
        type=str,
        default="zscore",
        help="高维输入归一化：none|minmax|zscore（默认 zscore）",
    )
    parser.add_argument(
        "--normalize-low",
        type=str,
        default="minmax",
        help="低维输入归一化：none|minmax|zscore（默认 minmax，贴近现有 boundary_analysis.py）",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="输出目录（默认：高维文件所在目录，即 data/<dataset>-features/）",
    )

    args = parser.parse_args()

    detector_name = (args.detector or "").lower().strip()
    try:
        detector = get_detector(detector_name)
    except Exception as e:
        raise SystemExit(str(e))

    high_path = os.path.abspath(args.high)
    if not os.path.exists(high_path):
        raise SystemExit(f"高维文件不存在: {high_path}")

    low_path = os.path.abspath(args.low) if args.low else None
    if low_path and (not os.path.exists(low_path)):
        raise SystemExit(f"低维文件不存在: {low_path}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else Path(high_path).parent.resolve()

    x_high_raw = load_array(high_path)
    x_high = normalize_array(x_high_raw, args.normalize_high)

    high_res: Optional[BoundaryResult] = None
    if not args.skip_high:
        k_high = int(args.k_high) if args.k_high is not None else int(args.k)
        high_res = detector(
            x_high,
            d=int(args.d),
            k=k_high,
            threshold_mode=args.threshold_mode,
            threshold_ratio=float(args.threshold_ratio),
        )

        # 高维输出 tag 固定为 features
        high_boundary_path, high_normal_path = save_points(
            out_dir,
            tag="features",
            detector_name=detector_name,
            x=x_high_raw,
            result=high_res,
        )
        print(
            f"[高维] N={x_high_raw.shape[0]} D={x_high_raw.shape[1]} | 边界={len(high_res.boundary_indices)} | 正常={len(high_res.normal_indices)}"
        )
        print(f"  已保存: {high_boundary_path}")
        print(f"  已保存: {high_normal_path}")
    else:
        print(f"[高维] 已跳过（--skip-high）")

    if low_path and (not args.skip_low):
        x_low_raw = load_array(low_path)
        if x_low_raw.shape[0] != x_high_raw.shape[0]:
            raise SystemExit(
                f"高维与低维样本数不一致：high N={x_high_raw.shape[0]} vs low N={x_low_raw.shape[0]}\n"
                "请确保低维 features 是由同一批高维特征按同一顺序降维得到。"
            )
        x_low = normalize_array(x_low_raw, args.normalize_low)

        k_low = int(args.k_low) if args.k_low is not None else int(args.k)
        low_res = detector(
            x_low,
            d=int(args.d),
            k=k_low,
            threshold_mode=args.threshold_mode,
            threshold_ratio=float(args.threshold_ratio),
        )

        low_method = infer_method_from_path(low_path)
        low_boundary_path, low_normal_path = save_points(
            out_dir,
            tag=low_method,
            detector_name=detector_name,
            x=x_low_raw,
            result=low_res,
        )
        print(f"[低维] N={x_low_raw.shape[0]} D={x_low_raw.shape[1]} | 边界={len(low_res.boundary_indices)} | 正常={len(low_res.normal_indices)}")
        print(f"  已保存: {low_boundary_path}")
        print(f"  已保存: {low_normal_path}")
    elif low_path and args.skip_low:
        print(f"[低维] 已跳过（--skip-low）")


if __name__ == "__main__":
    main()
