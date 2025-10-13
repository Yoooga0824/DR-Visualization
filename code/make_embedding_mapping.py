import numpy as np
import os
import json
import argparse
from pathlib import Path
from typing import Optional
import re


def scan_features_dirs(project_root: str) -> list:
    data_root = os.path.join(project_root, 'data')
    candidates = []
    if os.path.isdir(data_root):
        for name in os.listdir(data_root):
            p = os.path.join(data_root, name)
            if os.path.isdir(p) and name.endswith('-features'):
                candidates.append(os.path.abspath(p))
    return sorted(candidates)


def choose_from_list(title: str, options: list, allow_multi: bool = False, default_idxs=None) -> list:
    if default_idxs is None:
        default_idxs = [0]
    print(f"\n{title}")
    for i, opt in enumerate(options):
        print(f"  [{i+1}] {opt}")
    while True:
        sel = input("请输入编号" + ("（多选用逗号，如1,3；回车默认）" if allow_multi else "（单选；回车默认）") + ": ").strip()
        if not sel:
            return [options[i] for i in default_idxs if 0 <= i < len(options)]
        parts = [s.strip() for s in sel.split(',') if s.strip()]
        if all(p.isdigit() for p in parts):
            idxs = [int(p) - 1 for p in parts]
            chosen = [options[i] for i in idxs if 0 <= i < len(options)]
            if chosen:
                return chosen
        print("输入有误，请重试。")


def infer_raw_dirs(features_dir: str, dataset_name: str):
    raw_root = os.path.abspath(os.path.join(features_dir, '..', 'raw'))
    normal_candidates = [
        os.path.join(raw_root, dataset_name, dataset_name),
        os.path.join(raw_root, dataset_name),
        os.path.join(raw_root, 'Hands', 'Hands'),
        os.path.join(raw_root, 'Hands'),
    ]
    anom_candidates = [
        os.path.join(raw_root, f'Anomalous_{dataset_name}'),
        os.path.join(raw_root, 'Anomalous_Hands'),
    ]
    normal_candidates = [d for d in normal_candidates if os.path.isdir(d)]
    anom_candidates = [d for d in anom_candidates if os.path.isdir(d)]
    return raw_root, normal_candidates, anom_candidates


def scan_prefixes(features_dir: str) -> list:
    import re
    pattern = re.compile(r'^(anomalous_)?([A-Za-z0-9_]+)_embedding\.npy$')
    prefixes = set()
    for fname in os.listdir(features_dir):
        m = pattern.match(fname)
        if m and not fname.startswith('anomalous_'):
            prefixes.add(m.group(2))
    return sorted(prefixes)


def detect_default_glob(dir_path: str, prefer: Optional[list] = None) -> str:
    """
    在目录中自动检测常见图片扩展名，返回具有最多文件数量的 glob 模式。
    prefer: 优先顺序（扩展名列表，含点），例如 ['.png', '.jpg', '.jpeg']。
    """
    if prefer is None:
        prefer = ['.png', '.jpg', '.jpeg']
    dirp = Path(dir_path)
    counts = {}
    for ext in prefer:
        cnt = len(list(dirp.glob(f"*{ext}")))
        counts[ext] = cnt
    # 若全为 0，则回退到通配所有常见格式
    best_ext = max(counts, key=lambda k: counts[k]) if counts else '.png'
    if counts.get(best_ext, 0) == 0:
        return '*.*'
    return f"*{best_ext}"


def natural_key(s: str):
    """用于自然排序的 key，将连续数字按整数比较，其余按不区分大小写字符串比较。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def generate_mapping(prefix: str,
                     features_dir: str,
                     normal_dir: str,
                     anom_dir: str,
                     results_dir: str,
                     use_filter: bool,
                     anom_list_path: Optional[str],
                     normal_glob: str,
                     anom_glob: str,
                     natural_sort: bool = True):
    dataset_name = os.path.basename(features_dir).replace('-features', '')
    os.makedirs(results_dir, exist_ok=True)

    # embedding 路径
    normal_emb_path = os.path.join(features_dir, f'{prefix}_embedding.npy')
    anom_emb_path = os.path.join(features_dir, f'anomalous_{prefix}_embedding.npy')
    if not os.path.exists(normal_emb_path):
        raise FileNotFoundError(f"未找到正常 embedding: {normal_emb_path}")
    if not os.path.exists(anom_emb_path):
        raise FileNotFoundError(f"未找到异常 embedding: {anom_emb_path}")

    # failed_images 路径
    default_fail_path = os.path.join(features_dir, 'failed_images.txt')
    prefix_fail_path = os.path.join(features_dir, f'failed_images_{prefix}.txt')
    fail_path = prefix_fail_path if os.path.exists(prefix_fail_path) else default_fail_path

    # 异常样本 failed 过滤清单（优先前缀，其次默认）
    anom_default_fail_path = os.path.join(features_dir, 'anomalous_failed_images.txt')
    anom_prefix_fail_path = os.path.join(features_dir, f'anomalous_failed_images_{prefix}.txt')
    anom_fail_path = anom_prefix_fail_path if os.path.exists(anom_prefix_fail_path) else anom_default_fail_path

    # 1. 读取 failed_images（可禁用，支持 per-prefix）
    if use_filter:
        if os.path.exists(fail_path):
            with open(fail_path, 'r', encoding='utf-8') as f:
                failed = set(line.strip() for line in f if line.strip())
        else:
            failed = set()
    else:
        failed = set()

    # 2. 筛选未失败图片（使用可配置的匹配规则）
    if natural_sort:
        all_normal_images = sorted(list(Path(normal_dir).glob(normal_glob)), key=lambda p: natural_key(p.name))
    else:
        all_normal_images = sorted(list(Path(normal_dir).glob(normal_glob)))
    normal_images = [img for img in all_normal_images if img.name not in failed]

    # 3. 加载 embedding
    normal_embedding = np.load(normal_emb_path)
    if len(normal_images) != len(normal_embedding):
        raise ValueError(
            f"过滤后正常图片数({len(normal_images)})应和embedding数({len(normal_embedding)})严格一致！\n"
            f"- normal_dir: {normal_dir}\n- 匹配规则: {normal_glob}\n- 可尝试改用 --no-filter 或调整匹配规则。"
        )

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
    if natural_sort:
        anom_images = sorted(list(Path(anom_dir).glob(anom_glob)), key=lambda p: natural_key(p.name))
    else:
        anom_images = sorted(list(Path(anom_dir).glob(anom_glob)))

    # 4.1 异常样本 failed 过滤（可禁用、支持 per-prefix；当指定 anom_list_path 时跳过）
    if use_filter and (not anom_list_path):
        if os.path.exists(anom_fail_path):
            with open(anom_fail_path, 'r', encoding='utf-8') as f:
                anom_failed = set(line.strip() for line in f if line.strip())
            anom_images = [img for img in anom_images if img.name not in anom_failed]

    # 4.2 如果提供异常图片清单，则按清单精确选择与排序
    if anom_list_path:
        list_path = Path(anom_list_path)
        if not list_path.exists():
            raise FileNotFoundError(f"指定的异常图片清单不存在: {list_path}")
        by_name = {p.name: p for p in anom_images}
        selected = []
        missing = []
        with open(list_path, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue
                p = None
                if os.path.isabs(name):
                    p = Path(name)
                    if not p.exists():
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


def main():
    # 参数化前缀和所有路径（兼容命令行直传）
    parser = argparse.ArgumentParser(description='生成 embedding_to_image_mapping.json，支持多降维方法和多数据集（缺省进入交互式选择）')
    parser.add_argument('--prefix', type=str, help='降维方法前缀，如 TSNE、NeuralTSNE、UMAP；支持多个用逗号分隔')
    parser.add_argument('--features-dir', type=str, default=None, help='features 目录（如 data/Hands-features），自动/交互推断')
    parser.add_argument('--raw-dir', type=str, default=None, help='原始图片目录（如 data/raw/Hands/Hands）')
    parser.add_argument('--anom-dir', type=str, default=None, help='异常图片目录（如 data/raw/Anomalous_Hands 或 data/raw/Anomalous_yalefaces）')
    parser.add_argument('--results-dir', type=str, default=None, help='结果保存目录（如 results/Hands-results）')
    parser.add_argument('--no-filter', action='store_true', help='不使用 failed_images 过滤（同时影响正常与异常样本）')
    parser.add_argument('--anom-list', type=str, help='指定异常图片清单文件，用于精确匹配与排序异常图片')
    parser.add_argument('--normal-glob', type=str, help='正常图片匹配规则（如 Hand_*.jpg、*.png）；缺省自动从目录检测')
    parser.add_argument('--anom-glob', type=str, help='异常图片匹配规则（如 *.png）；缺省自动从目录检测')
    parser.add_argument('--no-natural-sort', action='store_true', help='禁用自然排序（数字按字典序而非数值排序）')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # 1) 选择 features 目录（数据集）
    if args.features_dir:
        features_dir = os.path.abspath(args.features_dir)
        if not os.path.isdir(features_dir):
            raise RuntimeError(f"指定的 features 目录不存在: {features_dir}")
    else:
        candidates = scan_features_dirs(project_root)
        if not candidates:
            raise RuntimeError(f"未在 {os.path.join(project_root, 'data')} 下找到任何 *-features 目录！")
        if len(candidates) == 1:
            features_dir = candidates[0]
            print(f"未指定 --features-dir，自动选择: {features_dir}")
        else:
            chosen = choose_from_list("检测到多个数据集目录，请选择：", candidates, allow_multi=False, default_idxs=[0])
            features_dir = chosen[0]

    dataset_name = os.path.basename(features_dir).replace('-features', '')

    # 2) 选择降维方法前缀（支持多选）
    if args.prefix:
        prefixes = [p.strip() for p in args.prefix.split(',') if p.strip()]
    else:
        scanned = scan_prefixes(features_dir)
        if not scanned:
            raise RuntimeError(f"在 {features_dir} 未找到任何 *_embedding.npy！")
        prefixes = choose_from_list("可用降维方法：", scanned, allow_multi=True, default_idxs=list(range(len(scanned))))

    # 3) 确认/选择原始与异常图片目录
    raw_root, normal_candidates, anom_candidates = infer_raw_dirs(features_dir, dataset_name)

    # 原始目录
    if args.raw_dir:
        normal_dir = os.path.abspath(args.raw_dir)
        if not os.path.isdir(normal_dir):
            raise RuntimeError(f"指定的原始图片目录不存在: {normal_dir}")
    else:
        if not normal_candidates:
            print(f"未在 {raw_root} 下找到明显的原始图片目录候选。")
            while True:
                manual = input("请手动输入原始图片目录路径：").strip('"')
                if manual and os.path.isdir(manual):
                    normal_dir = os.path.abspath(manual)
                    break
                print("路径无效，请重试。")
        elif len(normal_candidates) == 1:
            normal_dir = normal_candidates[0]
            print(f"自动选择原始图片目录: {normal_dir}")
        else:
            normal_dir = choose_from_list("检测到多个原始图片目录，请选择：", normal_candidates, allow_multi=False, default_idxs=[0])[0]

    # 异常目录
    if args.anom_dir:
        anom_dir = os.path.abspath(args.anom_dir)
        if not os.path.isdir(anom_dir):
            raise RuntimeError(f"指定的异常图片目录不存在: {anom_dir}")
    else:
        if not anom_candidates:
            print(f"未在 {raw_root} 下找到明显的异常图片目录候选。")
            while True:
                manual = input("请手动输入异常图片目录路径：").strip('"')
                if manual and os.path.isdir(manual):
                    anom_dir = os.path.abspath(manual)
                    break
                print("路径无效，请重试。")
        elif len(anom_candidates) == 1:
            anom_dir = anom_candidates[0]
            print(f"自动选择异常图片目录: {anom_dir}")
        else:
            anom_dir = choose_from_list("检测到多个异常图片目录，请选择：", anom_candidates, allow_multi=False, default_idxs=[0])[0]

    # 4) 选择输出目录
    if args.results_dir:
        results_dir = os.path.abspath(args.results_dir)
    else:
        results_root = os.path.abspath(os.path.join(features_dir, '..', '..', 'results'))
        default_results = os.path.join(results_root, f'{dataset_name}-results')
        print(f"\n建议输出目录: {default_results}")
        override = input("如需修改，请输入新路径；直接回车使用建议路径：").strip('"')
        results_dir = os.path.abspath(override) if override else default_results
    os.makedirs(results_dir, exist_ok=True)

    # 5) 正常/异常图片匹配规则
    if args.normal_glob:
        normal_glob = args.normal_glob
    else:
        # 对 Hands 提供 Hand_*.jpg 的优先建议；否则自动检测扩展名
        if dataset_name.lower() == 'hands' and len(list(Path(normal_dir).glob('Hand_*.jpg'))) > 0:
            guess = 'Hand_*.jpg'
        else:
            guess = detect_default_glob(normal_dir)
        print(f"\n正常图片匹配规则（glob）：建议 {guess}")
        entered = input("如需修改，请输入新的匹配（例如 *.png）；直接回车使用建议：").strip()
        normal_glob = entered if entered else guess

    if args.anom_glob:
        anom_glob = args.anom_glob
    else:
        anom_guess = detect_default_glob(anom_dir)
        print(f"异常图片匹配规则（glob）：建议 {anom_guess}")
        entered = input("如需修改，请输入新的匹配（例如 *.png）；直接回车使用建议：").strip()
        anom_glob = entered if entered else anom_guess

    # 6) 过滤与异常清单
    use_filter = not args.no_filter
    if args.no_filter is False:  # 仅当未通过参数明确禁用时，给一次交互确认
        ans = input("是否使用 failed_images 过滤？(Y/n)：").strip().lower()
        if ans == 'n':
            use_filter = False
    anom_list_path = args.anom_list
    if not anom_list_path:
        ans = input("是否指定异常图片清单进行精确匹配和排序？(y/N)，输入路径或直接回车：").strip('"')
        if ans and ans.lower() != 'n' and os.path.exists(ans):
            anom_list_path = ans

    natural_sort = not args.no_natural_sort

    # 7) 逐个前缀生成
    print("\n开始生成映射...")
    for p in prefixes:
        print(f"\n=== 处理方法: {p} ===")
        generate_mapping(prefix=p,
                         features_dir=features_dir,
                         normal_dir=normal_dir,
                         anom_dir=anom_dir,
                         results_dir=results_dir,
                         use_filter=use_filter,
                         anom_list_path=anom_list_path,
                         normal_glob=normal_glob,
                         anom_glob=anom_glob,
                         natural_sort=natural_sort)
    print("\n全部映射已生成完成。")


if __name__ == '__main__':
    main()