"""extract_image_features.py

通用：从原始图像批量提取高维特征（feature embeddings），并保存为 .npy。

设计目标（结合本仓库工作流）：
- 输入：一个或多个图片目录（常见约定：data/raw/<dataset>/；可选再合并 data/raw/Anomalous_<dataset>/）
- 输出：data/<dataset>-features/ 下的
    - features.npy（单一特征矩阵，不区分 normal/anomalous）
    - files.txt（记录参与特征提取的文件路径顺序，用于严格对齐）
    - failed_images.txt（记录读取失败的文件名，兼容 make_embedding_mapping.py 的过滤机制）

注意：
- 本脚本只做“图像 -> 高维特征”。降维（TSNE/UMAP/PCA）不在本脚本内。
- 为保证后续 embedding 与图片严格对齐，务必确保：
  1) 本脚本的文件排序规则稳定；
  2) 你在云端/其它环境做降维时使用的特征矩阵行顺序不被打乱；
  3) 或者始终使用本脚本输出的 files_*.txt 作为权威顺序。

依赖：torch, torchvision, pillow, numpy

示例（Windows PowerShell）：
    # 提取 Hands 数据集的高维特征（默认读取 data/raw/Hands，并自动合并 data/raw/Anomalous_Hands 若存在）
  py code/extract_image_features.py --dataset Hands --model resnet18 --batch-size 64

  # 指定自定义目录与输出目录
  py code/extract_image_features.py --normal-dir D:/data/myset --out-dir D:/DR-Visualization/data/myset-features
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def natural_key(name: str) -> list:
    """自然排序 key：'img2.png' < 'img10.png'。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def iter_images(root: Path, *, recursive: bool, exts: Sequence[str]) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []

    exts_norm = set(e.lower().lstrip(".") for e in exts)
    paths: List[Path] = []

    it: Iterable[Path]
    if recursive:
        it = root.rglob("*")
    else:
        it = root.glob("*")

    for p in it:
        if not p.is_file():
            continue
        suf = p.suffix.lower().lstrip(".")
        if suf in exts_norm:
            paths.append(p)

    # 稳定排序：先自然排序文件名，再用完整路径兜底
    paths.sort(key=lambda x: (natural_key(x.name), str(x).lower()))
    return paths


def _require_torch():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "缺少依赖 torch/torchvision，无法提取特征。\n"
            "请先安装：\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
            "或（CPU 版本）：\n"
            "  pip install torch torchvision\n"
            f"原始错误：{e}"
        )


def build_model(model_name: str):
    _require_torch()
    import torch
    import torchvision

    name = model_name.lower().strip()

    if name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        preprocess = weights.transforms()
        out_dim = 512
    elif name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        model = torchvision.models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        preprocess = weights.transforms()
        out_dim = 2048
    else:
        raise SystemExit(f"不支持的 --model: {model_name}（当前支持：resnet18, resnet50）")

    model.eval()
    return model, preprocess, out_dim


def choose_device(device: str) -> str:
    _require_torch()
    import torch

    d = device.lower().strip()
    if d in ("auto", ""):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d == "cuda":
        if not torch.cuda.is_available():
            print("[提示] 未检测到 CUDA，自动改用 CPU。")
            return "cpu"
        return "cuda"
    if d == "cpu":
        return "cpu"
    return d


def load_image_as_tensor(img_path: Path, preprocess):
    from PIL import Image

    with Image.open(img_path) as im:
        # 统一转 RGB，兼容灰度/L、RGBA、P 等模式
        if im.mode != "RGB":
            im = im.convert("RGB")
        return preprocess(im)


def extract_features(
    paths: Sequence[Path],
    *,
    model,
    preprocess,
    device: str,
    batch_size: int,
) -> Tuple[np.ndarray, List[Path], List[str]]:
    """返回：features[N, D]、成功列表、失败文件名列表（只保留 name，便于后续过滤）。"""
    _require_torch()
    import torch

    ok_paths: List[Path] = []
    failed_names: List[str] = []

    # 先逐个尝试读取/预处理，保证失败样本被剔除且可记录
    tensors: List[torch.Tensor] = []
    for p in paths:
        try:
            t = load_image_as_tensor(p, preprocess)
            tensors.append(t)
            ok_paths.append(p)
        except Exception:
            failed_names.append(p.name)

    if not ok_paths:
        return np.zeros((0, 0), dtype=np.float32), [], failed_names

    model = model.to(device)

    feats: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i : i + batch_size], dim=0).to(device)
            out = model(batch)
            out = out.detach().float().cpu().numpy()
            feats.append(out.astype(np.float32, copy=False))

    features = np.concatenate(feats, axis=0)
    return features, ok_paths, failed_names


def write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="通用图像特征提取：图片 -> 高维特征 .npy")
    parser.add_argument("--dataset", type=str, default=None, help="数据集名（默认推断目录 data/raw/<dataset> 与输出 data/<dataset>-features）")
    parser.add_argument("--raw-root", type=str, default=None, help="raw 根目录，默认 <repo>/data/raw")
    parser.add_argument("--image-dir", type=str, default=None, help="图片目录（覆盖 --dataset 推断）。可用多个目录请用 --dirs")
    parser.add_argument("--normal-dir", type=str, default=None, help="兼容旧参数：等价于 --image-dir")
    parser.add_argument("--anom-dir", type=str, default=None, help="可选：额外合并一个目录（历史上用于 anomalous；现仅作为第二输入目录，不再拆分输出）")
    parser.add_argument("--dirs", type=str, nargs='*', default=None, help="可选：一次性传入多个图片目录（会合并后提特征）")
    parser.add_argument("--out-dir", type=str, default=None, help="输出 features 目录（覆盖 --dataset 推断）")

    parser.add_argument("--model", type=str, default="resnet18", help="特征提取 backbone（resnet18/resnet50）")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小（CPU 可调小）")

    parser.add_argument("--exts", type=str, default="jpg,jpeg,png,bmp,webp,tif,tiff", help="允许的图片扩展名（逗号分隔）")
    parser.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    parser.add_argument("--no-auto-merge-anom", action="store_true", help="当 --dataset 指定时，默认会自动合并 data/raw/Anomalous_<dataset>（若存在）；传此参数可禁用")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    raw_root = Path(args.raw_root) if args.raw_root else (repo_root / "data" / "raw")

    dataset = args.dataset

    # 兼容旧参数：--normal-dir 等价于 --image-dir
    image_dir = args.image_dir or args.normal_dir

    # 收集输入目录（合并后提特征）
    input_dirs: List[Path] = []
    if args.dirs and len(args.dirs) > 0:
        input_dirs.extend([Path(d).resolve() for d in args.dirs if d])
    if image_dir:
        input_dirs.append(Path(image_dir).resolve())
    if args.anom_dir:
        input_dirs.append(Path(args.anom_dir).resolve())

    if not input_dirs:
        if not dataset:
            raise SystemExit("需要提供 --dataset 或 --image-dir/--normal-dir 或 --dirs")
        input_dirs.append((raw_root / dataset).resolve())

        # 自动合并 Anomalous_<dataset>（若存在且未禁用）
        if not args.no_auto_merge_anom:
            auto_anom = (raw_root / f"Anomalous_{dataset}").resolve()
            if auto_anom.exists() and auto_anom.is_dir():
                input_dirs.append(auto_anom)

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        if not dataset:
            raise SystemExit("未提供 --out-dir 且无法从 --dataset 推断输出目录")
        out_dir = (repo_root / "data" / f"{dataset}-features").resolve()

    exts = [e.strip() for e in args.exts.split(",") if e.strip()]

    print("输入目录：")
    for d in input_dirs:
        print(f" - {d}")
    print(f"输出目录: {out_dir}")

    all_paths: List[Path] = []
    for d in input_dirs:
        all_paths.extend(iter_images(d, recursive=args.recursive, exts=exts))
    # 全局稳定排序（跨目录也保持确定性）
    all_paths.sort(key=lambda x: (natural_key(x.name), str(x).lower()))
    if not all_paths:
        raise SystemExit("输入目录未找到任何图片。")

    device = choose_device(args.device)
    model, preprocess, out_dim = build_model(args.model)

    print(f"使用模型: {args.model} | 输出维度: {out_dim} | device: {device}")
    print(f"总图片数: {len(all_paths)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 单一集合（合并后）
    features, ok_paths, failed_names = extract_features(
        all_paths,
        model=model,
        preprocess=preprocess,
        device=device,
        batch_size=max(1, int(args.batch_size)),
    )

    if features.size == 0:
        raise SystemExit("所有图片读取失败，未生成任何特征。")

    # 保存：features + 顺序清单 + 失败清单
    # 命名规范：高维特征固定保存为 features.npy
    np.save(out_dir / "features.npy", features)
    write_lines(out_dir / "files.txt", [str(p) for p in ok_paths])
    write_lines(out_dir / "failed_images.txt", failed_names)

    print(f"已保存特征: {out_dir / 'features.npy'} | shape={features.shape}")
    if failed_names:
        print(f"读取失败: {len(failed_names)}（已写入 failed_images.txt）")

    print("完成：特征矩阵行顺序以 files.txt 为准，可用于云端降维时严格对齐。")


if __name__ == "__main__":
    main()
