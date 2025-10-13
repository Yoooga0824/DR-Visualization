import numpy as np
import os
import json
import argparse
from pathlib import Path


# 参数化前缀和所有路径
parser = argparse.ArgumentParser(description='生成 embedding_to_image_mapping.json，支持多降维方法和多数据集')
parser.add_argument('--prefix', type=str, default='TSNE', help='降维方法前缀，如 TSNE、NeuralTSNE、UMAP')
parser.add_argument('--features-dir', type=str, default=None, help='features 目录（如 data/Hands-features），自动推断数据集名')
parser.add_argument('--raw-dir', type=str, default=None, help='原始图片目录（如 data/raw/Hands/Hands），如未指定自动推断')
parser.add_argument('--anom-dir', type=str, default=None, help='异常图片目录（如 data/raw/Anomalous_Hands 或 data/raw/Anomalous_yalefaces），如未指定自动推断')
parser.add_argument('--results-dir', type=str, default=None, help='结果保存目录（如 results/Hands-results），如未指定自动推断')
parser.add_argument('--no-filter', action='store_true', help='不使用 failed_images 过滤（同时影响正常与异常样本）')
parser.add_argument('--anom-list', type=str, help='指定异常图片清单文件（每行一个文件名或绝对路径），用于精确匹配与排序异常图片')
args = parser.parse_args()
prefix = args.prefix


# 自动推断 features 目录
# 优先使用传入参数；否则在项目 data 目录下查找任意 *-features 目录：
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

# 原始图片目录
if args.raw_dir:
    hands_dir = args.raw_dir
else:
    # 常见命名兼容 Hands/Hands、yalefaces、等
    raw_root = os.path.abspath(os.path.join(features_dir, '..', 'raw'))
    hands_candidates = [
        os.path.join(raw_root, dataset_name, dataset_name),
        os.path.join(raw_root, dataset_name),
        os.path.join(raw_root, 'Hands', 'Hands'),
        os.path.join(raw_root, 'Hands'),
    ]
    hands_dir = next((d for d in hands_candidates if os.path.isdir(d)), None)
    if not hands_dir:
        raise RuntimeError(f'未找到原始图片目录，请通过 --raw-dir 指定，已尝试: {hands_candidates}')

# 异常图片目录
if args.anom_dir:
    anom_dir = args.anom_dir
else:
    anom_candidates = [
        os.path.join(raw_root, f'Anomalous_{dataset_name}'),
        os.path.join(raw_root, 'Anomalous_Hands'),
    ]
    anom_dir = next((d for d in anom_candidates if os.path.isdir(d)), None)
    if not anom_dir:
        raise RuntimeError(f'未找到异常图片目录，请通过 --anom-dir 指定，已尝试: {anom_candidates}')

# 结果保存目录
if args.results_dir:
    results_dir = args.results_dir
else:
    results_root = os.path.abspath(os.path.join(features_dir, '..', '..', 'results'))
    results_dir = os.path.join(results_root, f'{dataset_name}-results')
os.makedirs(results_dir, exist_ok=True)

# embedding 路径
normal_emb_path = os.path.join(features_dir, f'{prefix}_embedding.npy')
anom_emb_path   = os.path.join(features_dir, f'anomalous_{prefix}_embedding.npy')

# failed_images 路径
default_fail_path = os.path.join(features_dir, 'failed_images.txt')
prefix_fail_path = os.path.join(features_dir, f'failed_images_{prefix}.txt')
fail_path = prefix_fail_path if os.path.exists(prefix_fail_path) else default_fail_path

# 异常样本 failed 过滤清单（优先前缀，其次默认）
anom_default_fail_path = os.path.join(features_dir, 'anomalous_failed_images.txt')
anom_prefix_fail_path = os.path.join(features_dir, f'anomalous_failed_images_{prefix}.txt')
anom_fail_path = anom_prefix_fail_path if os.path.exists(anom_prefix_fail_path) else anom_default_fail_path

# 1. 读取 failed_images（可禁用，支持 per-prefix）
if args.no_filter:
    failed = set()
else:
    if os.path.exists(fail_path):
        with open(fail_path, 'r', encoding='utf-8') as f:
            failed = set(line.strip() for line in f if line.strip())
    else:
        failed = set()

# 2. 筛选未失败图片
all_normal_images = sorted(list(Path(hands_dir).glob("Hand_*.jpg")))
normal_images = [img for img in all_normal_images if img.name not in failed]


# 3. 加载 embedding
normal_embedding = np.load(normal_emb_path)
if len(normal_images) != len(normal_embedding):
    raise ValueError(f"过滤后正常图片数({len(normal_images)})应和embedding数({len(normal_embedding)})严格一致！")

normal_mapping = []
for i, (img, emb) in enumerate(zip(normal_images, normal_embedding)):
    normal_mapping.append({
        "index": i,
        "embedding_coords": emb.tolist(),
        "image_path": str(img),
        "features": []
    })

## 4. 异常图片（支持 failed 过滤）
anom_embedding = np.load(anom_emb_path)
anom_images = sorted(list(Path(anom_dir).glob("*.jpg")))

# 4.1 异常样本 failed 过滤（可禁用、支持 per-prefix；当指定 --anom-list 时跳过）
if (not args.no_filter) and (not args.anom_list):
    if os.path.exists(anom_fail_path):
        with open(anom_fail_path, 'r', encoding='utf-8') as f:
            anom_failed = set(line.strip() for line in f if line.strip())
        anom_images = [img for img in anom_images if img.name not in anom_failed]

# 4.2 如果提供异常图片清单，则按清单精确选择与排序
if args.anom_list:
    list_path = Path(args.anom_list)
    if not list_path.exists():
        raise FileNotFoundError(f"指定的异常图片清单不存在: {list_path}")
    # 构建基于文件名的查找表
    by_name = {p.name: p for p in anom_images}
    selected = []
    missing = []
    with open(list_path, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            p = None
            # 既支持绝对路径，也支持仅文件名
            if os.path.isabs(name):
                p = Path(name)
                if not p.exists():
                    # 回退按文件名匹配
                    p = by_name.get(Path(name).name)
            else:
                p = by_name.get(name)
            if p is None:
                missing.append(name)
            else:
                selected.append(p)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"清单中的异常图片在目录中未找到: {len(missing)} 个，例如: {preview} ...")
    anom_images = selected

if len(anom_images) != len(anom_embedding):
    raise ValueError(
        f"异常图片数({len(anom_images)})应和embedding数({len(anom_embedding)})严格一致！"
        f" 如使用了自定义样本集合，请通过 --anom-list 指定清单以精确匹配与排序。"
    )
anom_mapping = []
for i, (img, emb) in enumerate(zip(anom_images, anom_embedding)):
    anom_mapping.append({
        "index": len(normal_mapping)+i,
        "embedding_coords": emb.tolist(),
        "image_path": str(img),
        "features": []
    })

# 合并并保存
mapping_data = normal_mapping + anom_mapping

mapping_filename = f"embedding_to_image_mapping_{prefix}.json"
with open(os.path.join(results_dir, mapping_filename), "w", encoding="utf-8") as f:
    json.dump(mapping_data, f, indent=2, ensure_ascii=False)

print(f"{mapping_filename} 已生成！共{len(mapping_data)}条。保存于: {results_dir}")

from collections import Counter

paths = [item['image_path'] for item in mapping_data]
counter = Counter(paths)
repeats = [(p, c) for p, c in counter.items() if c > 1]
print(f"总点数：{len(paths)}")
print(f"唯一图片数：{len(set(paths))}")
print(f"出现多次的图片及重复次数：{repeats[:10]}")