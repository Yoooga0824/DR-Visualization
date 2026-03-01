"""compute_embeddings.py

从高维特征矩阵（features.npy）计算低维特征（<降维方法>_features.npy）。

- 读取：features.npy（二维 [N, D]）
- 输出：<method>_features.npy（二维 [N, n_components]，默认 2）

输出文件命名遵循你的实验约定：
- <降维方法>_features.npy
其中降维方法例如：PCA、TSNE、UMAP。

示例：
    # 生成 PCA 2D features
    py code/compute_embeddings.py --features data/Hands-features/features.npy --method pca

  # 一次生成多个方法（逗号分隔）
    py code/compute_embeddings.py --features data/Hands-features/features.npy --methods pca,tsne

  # UMAP（需要安装 umap-learn）
    py code/compute_embeddings.py --features data/Hands-features/features.npy --method umap --umap-n-neighbors 30
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np

from dimred import get_reducer, list_methods


def load_features(path: str) -> np.ndarray:
    x = np.load(path)
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        raise SystemExit(f"features 必须是二维数组 [N, D]：{path}")
    return x


def save_features(out_dir: Path, prefix: str, z: np.ndarray) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}_features.npy"
    np.save(out_path, z.astype(np.float32, copy=False))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="从 features.npy 计算低维 <method>_features.npy（PCA/TSNE/UMAP）")
    parser.add_argument("--features", required=True, help="高维特征 features.npy（二维 [N, D]）")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--method", type=str, default=None, help=f"单个降维方法：{list_methods()}")
    group.add_argument("--methods", type=str, default=None, help="多个方法，逗号分隔，例如 pca,tsne,umap")

    parser.add_argument("--n-components", type=int, default=2, help="输出维度（默认 2）")
    parser.add_argument("--out-dir", type=str, default=None, help="输出目录（默认与 features 同目录）")

    # TSNE 参数
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-n-iter", type=int, default=1000)
    parser.add_argument("--tsne-init", type=str, default="pca")

    # UMAP 参数
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-metric", type=str, default="euclidean")

    # 通用随机种子
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    features_path = os.path.abspath(args.features)
    if not os.path.exists(features_path):
        raise SystemExit(f"features 文件不存在：{features_path}")

    x = load_features(features_path)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else Path(features_path).parent.resolve()

    if args.methods:
        method_list = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    elif args.method:
        method_list = [args.method.strip().lower()]
    else:
        # 默认只跑 PCA，保证无额外依赖也可跑通
        method_list = ["pca"]

    for m in method_list:
        reducer = get_reducer(m)
        prefix = m.upper()
        print(f"\n=== 计算 {prefix} features ===")

        kwargs = {"n_components": int(args.n_components)}
        if m == "tsne":
            kwargs.update(
                {
                    "perplexity": float(args.tsne_perplexity),
                    "n_iter": int(args.tsne_n_iter),
                    "init": str(args.tsne_init),
                    "random_state": int(args.seed),
                }
            )
        elif m == "umap":
            kwargs.update(
                {
                    "n_neighbors": int(args.umap_n_neighbors),
                    "min_dist": float(args.umap_min_dist),
                    "metric": str(args.umap_metric),
                    "random_state": int(args.seed),
                }
            )
        else:
            # pca：可扩展其它参数
            kwargs.update({"random_state": int(args.seed)})

        z = reducer(x, **kwargs)
        if not isinstance(z, np.ndarray) or z.ndim != 2 or z.shape[0] != x.shape[0]:
            raise SystemExit(f"降维输出形状异常：期望 [N, k] 且 N={x.shape[0]}，实际 {getattr(z, 'shape', None)}")

        out_path = save_features(out_dir, prefix, z)
        print(f"已保存：{out_path} | shape={z.shape}")


if __name__ == "__main__":
    main()
