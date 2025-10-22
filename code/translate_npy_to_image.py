"""
translate_npy_to_image.py

用途：
- 将保存在 Numpy 数组（.npy）中的扁平化图像（如 MNIST 28x28=784）反向还原为灰度 PNG 图像；
- 支持同时处理“正常集”（normal）与“异常集”（anomalous），并分别输出到 data/raw/<dataset> 与 data/raw/Anomalous_<dataset>；
- 文件命名为 <index>.png，保证与数组顺序一一对应，便于后续映射与可视化。

示例（Windows PowerShell）：
    py code/translate_npy_to_image.py ^
        --dataset mnist ^
        --normal-npy d:/DR-Visualization/data/mnist-features/mnist_features.npy ^
        --anom-npy   d:/DR-Visualization/data/mnist-features/anomalous_mnist_features.npy
"""

import os
import math
import argparse
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image


def compute_side_length(dim: int) -> int:
    m = int(round(math.sqrt(dim)))
    if m * m != dim:
        raise ValueError(f"无法将长度为 {dim} 的扁平数组还原为正方形图像（非完全平方数）。")
    return m


def to_uint8_image(img: np.ndarray) -> np.ndarray:
    """
    将任意数值范围的二维数组转换为 8-bit 灰度图：
    - 若整体在 [0, 1]，直接乘 255
    - 若整体在 [0, 255]，裁剪到 [0,255]
    - 否则对每张图做 min-max 归一化到 [0,255]
    """
    img = img.astype(np.float32)
    min_val = float(np.min(img))
    max_val = float(np.max(img))
    if max_val <= 1.0 and min_val >= 0.0:
        arr = img * 255.0
    elif max_val <= 255.0 and min_val >= 0.0:
        arr = np.clip(img, 0.0, 255.0)
    else:
        if max_val > min_val:
            arr = (img - min_val) / (max_val - min_val) * 255.0
        else:
            arr = np.zeros_like(img, dtype=np.float32)
    return arr.astype(np.uint8)


def save_images_from_npy(npy_path: str, out_dir: str, *, transpose: bool = False) -> int:
    raw = np.load(npy_path, allow_pickle=True)
    data = None
    # 情况 A：0 维 ndarray，内部包裹 dict/list/ndarray
    if isinstance(raw, np.ndarray) and raw.ndim == 0:
        obj = raw.item()
        if isinstance(obj, dict):
            # 优先常见键 'data' 或 'X'
            if 'data' in obj:
                data = np.asarray(obj['data'])
            elif 'X' in obj:
                data = np.asarray(obj['X'])
            else:
                raise ValueError(f"{npy_path} 的字典对象缺少 'data' 或 'X' 键，无法提取图像数据。现有键：{list(obj.keys())}")
        elif isinstance(obj, (list, tuple)):
            try:
                data = np.stack([np.asarray(x).ravel() for x in obj], axis=0)
            except Exception as e:
                raise ValueError(f"{npy_path} 为 0 维对象列表，无法堆叠为二维矩阵：{e}")
        elif isinstance(obj, np.ndarray):
            data = obj
        else:
            raise ValueError(f"{npy_path} 的 0 维对象类型不受支持：{type(obj)}")
    # 情况 B：object 数组，每个元素是一张图
    elif isinstance(raw, np.ndarray) and raw.dtype == object:
        try:
            data = np.stack([np.asarray(x).ravel() for x in raw], axis=0)
        except Exception as e:
            raise ValueError(f"{npy_path} 为 object 数组，且无法堆叠为二维矩阵：{e}")
    # 情况 C：已是数值型二维矩阵
    elif isinstance(raw, np.ndarray):
        data = raw
    else:
        raise ValueError(f"{npy_path} 的类型不受支持：{type(raw)}")
    if data.ndim != 2:
        raise ValueError(f"{npy_path} 的维度为 {data.ndim}，期望为二维 [N, D]")
    n, d = data.shape
    m = compute_side_length(d)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for j in range(n):
        img_flat = data[j, :]
        img = img_flat.reshape(m, m)
        if transpose:
            img = np.transpose(img)
        img_uint8 = to_uint8_image(img)
        # 文件名直接用索引，保证与数组行号一一对应
        filepath = os.path.join(out_dir, f"{j}.png")
        Image.fromarray(img_uint8, mode="L").save(filepath)
    return n


def main():
    parser = argparse.ArgumentParser(description="将扁平化图像的 .npy 还原为 PNG 图像（保持索引顺序一一对应）")
    parser.add_argument('--dataset', type=str, default='mnist', help='数据集名称（决定默认输出目录名）')
    parser.add_argument('--normal-npy', type=str, required=True, help='正常集 .npy 路径（形如 [N, D]）')
    parser.add_argument('--anom-npy', type=str, required=False, help='异常集 .npy 路径（形如 [M, D]）')
    parser.add_argument('--out-normal', type=str, default=None, help='正常图输出目录（默认 data/raw/<dataset>）')
    parser.add_argument('--out-anom', type=str, default=None, help='异常图输出目录（默认 data/raw/Anomalous_<dataset>）')
    parser.add_argument('--transpose', action='store_true', help='是否在保存前对图像做转置（YaleFaces 需要）')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    raw_root = os.path.join(project_root, 'data', 'raw')
    out_normal = args.out_normal or os.path.join(raw_root, args.dataset)
    out_anom = args.out_anom or os.path.join(raw_root, f'Anomalous_{args.dataset}')

    print(f"正常集输入: {args.normal_npy}")
    print(f"正常图输出: {out_normal}")
    n_count = save_images_from_npy(args.normal_npy, out_normal, transpose=args.transpose)
    print(f"已保存正常图像 {n_count} 张 → {out_normal}")

    if args.anom_npy:
        print(f"异常集输入: {args.anom_npy}")
        print(f"异常图输出: {out_anom}")
        a_count = save_images_from_npy(args.anom_npy, out_anom, transpose=args.transpose)
        print(f"已保存异常图像 {a_count} 张 → {out_anom}")
    else:
        print("未提供 --anom-npy，跳过异常集。")

    print("完成：图像文件命名为 <index>.png，与 .npy 中的行索引一一对应。")


if __name__ == '__main__':
    main()