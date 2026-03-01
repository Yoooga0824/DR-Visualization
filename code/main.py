import os
import re
import subprocess
import argparse
from pathlib import Path

import glob
import numpy as np


# 自动遍历所有 features 目录（如 data/Hands-features, data/yalefaces-features 等）
FEATURES_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
FEATURES_DIRS_ALL = [d for d in glob.glob(os.path.join(FEATURES_ROOT, '*-features')) if os.path.isdir(d)]
if not FEATURES_DIRS_ALL:
    raise RuntimeError(f'未找到任何 *-features 目录，请检查 {FEATURES_ROOT} 目录结构！')
CODE_DIR = os.path.abspath(os.path.dirname(__file__))
RESULTS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

# 交互式选择数据集目录
def choose_features_dirs(features_dirs):
    print("\n可用数据集目录：")
    for i, d in enumerate(features_dirs):
        print(f"  [{i+1}] {os.path.basename(d)}  ({d})")
    while True:
        sel = input("请输入要处理的数据集编号（回车全选，支持多选如1,3）：").strip()
        if not sel:
            return features_dirs  # 全选
        try:
            idxs = [int(x)-1 for x in sel.split(',') if x.strip().isdigit()]
            chosen = [features_dirs[i] for i in idxs if 0 <= i < len(features_dirs)]
            if chosen:
                return chosen
        except Exception:
            pass
        print("输入有误，请重新输入！")


# 1. 搜索所有 features 目录下的 embedding 前缀（主流程已在后面实现，这里移除初始化代码）

def choose_method(method_prefixes):
    print("\n可用降维方法：")
    for i, p in enumerate(method_prefixes):
        print(f"  [{i+1}] {p}")
    while True:
        sel = input("请输入要生成的降维方法编号（回车全选，支持多选如1,3）：").strip()
        if not sel:
            return method_prefixes  # 全选
        try:
            idxs = [int(x)-1 for x in sel.split(',') if x.strip().isdigit()]
            chosen = [method_prefixes[i] for i in idxs if 0 <= i < len(method_prefixes)]
            if chosen:
                return chosen
        except Exception:
            pass
        print("输入有误，请重新输入！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, help='只处理指定降维方法前缀')
    parser.add_argument('--features-dir', type=str, help='只处理指定 features 目录（可多次指定）', nargs='*')
    args = parser.parse_args()

    # 选择数据集目录
    if args.features_dir and len(args.features_dir) > 0:
        features_dirs = [os.path.abspath(d) for d in args.features_dir if os.path.isdir(d)]
        if not features_dirs:
            print("未找到有效的 features 目录，自动切换为交互选择！")
            features_dirs = choose_features_dirs(FEATURES_DIRS_ALL)
    else:
        features_dirs = choose_features_dirs(FEATURES_DIRS_ALL)

    print(f"检测到数据集目录: {[os.path.basename(d) for d in features_dirs]}")

    # 搜索所有选中 features 目录下的方法前缀（优先 *_features.npy，兼容旧 *_embedding.npy）
    method_prefixes = set()
    dataset_names = {os.path.basename(d).replace('-features', '').lower() for d in features_dirs}
    patterns = [
        re.compile(r'^(anomalous_)?([A-Za-z0-9_]+)_features\.npy$'),
        re.compile(r'^(anomalous_)?([A-Za-z0-9_]+)_embedding\.npy$'),
    ]
    def is_low_dim_coords(path: str) -> bool:
        try:
            arr = np.load(path, mmap_mode='r')
            return isinstance(arr, np.ndarray) and arr.ndim == 2 and 2 <= arr.shape[1] <= 3
        except Exception:
            return False

    for features_dir in features_dirs:
        files = os.listdir(features_dir)
        for fname in files:
            for pattern in patterns:
                m = pattern.match(fname)
                if not m:
                    continue
                if fname.startswith('anomalous_'):
                    continue
                p = m.group(2)
                # 排除高维特征文件（旧：<dataset>_features.npy；新：features.npy 不会匹配这里）
                if p.lower() in dataset_names:
                    continue
                full = os.path.join(features_dir, fname)
                if not is_low_dim_coords(full):
                    continue
                method_prefixes.add(p)
                break
    method_prefixes = sorted(method_prefixes)

    # 选择降维方法
    if args.prefix:
        method_prefixes = [args.prefix]
    else:
        method_prefixes = choose_method(method_prefixes)

    print(f"检测到降维方法: {method_prefixes}")

    for prefix in method_prefixes:
        print(f"\n==== 处理方法: {prefix} ====")
        # 针对每个 features 目录都执行 mapping 和 HTML 生成
        for features_dir in features_dirs:
            # 检查该 features 目录下是否有该 prefix 的坐标文件（优先 *_features.npy）
            feat_path = os.path.join(features_dir, f'{prefix}_features.npy')
            emb_path = os.path.join(features_dir, f'{prefix}_embedding.npy')
            if (not os.path.exists(feat_path)) and (not os.path.exists(emb_path)):
                print(f"[跳过] {features_dir} 不存在 {prefix}_features.npy / {prefix}_embedding.npy")
                continue
            print(f"-- 数据集目录: {features_dir}")
            # 2. 生成 mapping
            mapping_cmd = [
                'python', os.path.join(CODE_DIR, 'make_embedding_mapping.py'),
                '--prefix', prefix,
                '--features-dir', features_dir
            ]
            print(' '.join(mapping_cmd))
            subprocess.run(mapping_cmd, check=True)

            # 3. 生成 HTML
            analysis_cmd = [
                'python', os.path.join(CODE_DIR, 'boundary_analysis.py'),
                '--prefix', prefix,
                '--features-dir', features_dir,
                '--img-base', 'http://localhost:5678'
            ]
            print(' '.join(analysis_cmd))
            subprocess.run(analysis_cmd, check=True)

    print("\n全部方法处理完成！每个方法都已生成独立 mapping 和 HTML 文件。")
