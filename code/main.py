from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
RAW_ROOT_DEFAULT = DATA_ROOT / "raw"

CODE_DIR = Path(__file__).resolve().parent


@dataclass
class DatasetPaths:
    dataset: str
    features_dir: Path
    raw_dirs: List[Path]
    results_dir: Path


def natural_key(s: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def choose_from_list(title: str, options: Sequence[str], *, allow_multi: bool, default_idxs: Sequence[int]) -> List[str]:
    if not options:
        raise ValueError("options 为空")

    print(f"\n{title}")
    for i, opt in enumerate(options):
        print(f"  [{i + 1}] {opt}")

    defaults = [options[i] for i in default_idxs if 0 <= i < len(options)]
    if defaults:
        print("默认：" + ("、".join(defaults)))

    while True:
        prompt = "请输入编号"
        if allow_multi:
            prompt += "（多选用逗号，如1,3；回车默认）："
        else:
            prompt += "（单选；回车默认）："

        sel = input(prompt).strip()
        if not sel:
            return [options[i] for i in default_idxs if 0 <= i < len(options)]

        parts = [p.strip() for p in sel.split(",") if p.strip()]
        if all(p.isdigit() for p in parts):
            idxs = [int(p) - 1 for p in parts]
            chosen = [options[i] for i in idxs if 0 <= i < len(options)]
            if chosen:
                return chosen

        print("输入有误，请重试。")


def ask_yes_no(question: str, default_yes: bool = True) -> bool:
    suffix = "(Y/n)" if default_yes else "(y/N)"
    ans = input(f"{question} {suffix}: ").strip().lower()
    if not ans:
        return default_yes
    return ans in ("y", "yes")


def scan_features_dirs() -> List[Path]:
    out: List[Path] = []
    if DATA_ROOT.exists():
        for p in DATA_ROOT.iterdir():
            if p.is_dir() and p.name.endswith("-features"):
                out.append(p.resolve())
    out.sort(key=lambda p: natural_key(p.name))
    return out


def scan_raw_dirs(raw_root: Path) -> List[Path]:
    if not raw_root.exists():
        return []
    out = [p.resolve() for p in raw_root.iterdir() if p.is_dir()]
    out.sort(key=lambda p: natural_key(p.name))
    return out


def is_low_dim_coords(path: Path) -> bool:
    try:
        arr = np.load(str(path), mmap_mode="r")
        return isinstance(arr, np.ndarray) and arr.ndim == 2 and 2 <= arr.shape[1] <= 3
    except Exception:
        return False


def find_existing_methods(features_dir: Path) -> List[str]:
    """返回当前 features_dir 下已有的低维方法前缀（只统计 2D/3D 坐标）。"""
    prefixes: set[str] = set()
    for p in features_dir.iterdir():
        if not p.is_file():
            continue
        m = re.match(r"^([A-Za-z0-9_]+)_(features|embedding)\.npy$", p.name)
        if not m:
            continue
        prefix = m.group(1)
        if prefix.lower() in ("anomalous", "features"):
            continue
        if is_low_dim_coords(p):
            prefixes.add(prefix.upper())
    return sorted(prefixes)


def run_py(script: Path, args: List[str]) -> None:
    cmd = [sys.executable, str(script)] + args
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_features_file(features_dir: Path, dataset: str) -> Path:
    """确保按新约定存在 features.npy；若只有旧的 <dataset>_features.npy 则引导复制。"""
    target = features_dir / "features.npy"
    if target.exists():
        return target

    legacy = features_dir / f"{dataset}_features.npy"
    if legacy.exists():
        print(f"\n[兼容] 检测到旧命名高维特征：{legacy}")
        if ask_yes_no("是否复制为 features.npy 以符合新命名？", default_yes=True):
            # 直接复制文件（避免 load/save 占内存）
            target.write_bytes(legacy.read_bytes())
            print(f"已生成：{target}")
            return target
        print("将继续使用旧文件路径，但后续建议统一为 features.npy。")
        return legacy

    raise FileNotFoundError(f"未找到高维特征文件：{target}（也未找到兼容的 {legacy}）")


def resolve_low_features_path(features_dir: Path, method_prefix: str) -> Path:
    """优先 <METHOD>_features.npy，否则兼容 <METHOD>_embedding.npy。"""
    p1 = features_dir / f"{method_prefix}_features.npy"
    if p1.exists():
        return p1
    p2 = features_dir / f"{method_prefix}_embedding.npy"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"未找到低维文件：{p1}（或兼容旧 {p2}）")


def get_dimred_methods() -> List[str]:
    sys.path.insert(0, str(CODE_DIR))
    from dimred import list_methods  # type: ignore

    return [m.upper() for m in list_methods()]


def get_detectors() -> List[str]:
    sys.path.insert(0, str(CODE_DIR))
    from detection import list_detectors  # type: ignore

    return list_detectors()


def can_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def filter_dimred_by_deps(methods: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """返回（可用方法, 不可用方法及原因提示）。"""
    available: List[str] = []
    unavailable: List[Tuple[str, str]] = []

    has_sklearn = can_import("sklearn")
    has_umap = can_import("umap")

    for m in methods:
        mu = m.upper()
        if mu == "TSNE" and not has_sklearn:
            unavailable.append((mu, "缺少 scikit-learn（安装：pip install scikit-learn）"))
            continue
        if mu == "UMAP" and not has_umap:
            unavailable.append((mu, "缺少 umap-learn（安装：pip install umap-learn）"))
            continue
        available.append(mu)

    return available, unavailable


def filter_detectors_by_deps(detectors: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    available: List[str] = []
    unavailable: List[Tuple[str, str]] = []

    has_sklearn = can_import("sklearn")
    has_scipy = can_import("scipy")

    for d in detectors:
        dl = d.lower()
        if dl in ("bdlle", "knn_distance") and not has_scipy:
            unavailable.append((d, "缺少 scipy（安装：pip install scipy）"))
            continue
        if dl == "ocsvm" and not has_sklearn:
            unavailable.append((d, "缺少 scikit-learn（安装：pip install scikit-learn）"))
            continue
        available.append(d)

    return available, unavailable


def feature_extraction_available() -> Tuple[bool, str]:
    # extract_image_features.py 依赖 torch/torchvision
    if not can_import("torch"):
        return False, "缺少 torch（安装：pip install torch）"
    if not can_import("torchvision"):
        return False, "缺少 torchvision（安装：pip install torchvision）"
    return True, ""


def pick_datasets_interactive() -> List[DatasetPaths]:
    features_dirs = scan_features_dirs()
    raw_dirs = scan_raw_dirs(RAW_ROOT_DEFAULT)

    ds_options: List[str] = []
    # 优先从 features 目录推断数据集
    for fd in features_dirs:
        ds_options.append(fd.name.replace("-features", ""))
    # 再补充 raw 目录中可能存在但没有 features 的数据集
    for rd in raw_dirs:
        name = rd.name
        if name.lower().startswith("anomalous_"):
            continue
        if name not in ds_options:
            ds_options.append(name)

    if not ds_options:
        raise RuntimeError("未检测到任何数据集：data/*-features 或 data/raw/* 均为空")

    ds_options.sort(key=natural_key)
    chosen = choose_from_list("选择数据集（可多选）：", ds_options, allow_multi=True, default_idxs=list(range(len(ds_options))))

    out: List[DatasetPaths] = []
    for ds in chosen:
        features_dir = (DATA_ROOT / f"{ds}-features").resolve()
        results_dir = (REPO_ROOT / "results" / f"{ds}-results").resolve()

        # raw dirs：默认 raw/<ds>，并可选自动合并 raw/Anomalous_<ds>
        raw_main = (RAW_ROOT_DEFAULT / ds).resolve()
        raw_anom = (RAW_ROOT_DEFAULT / f"Anomalous_{ds}").resolve()
        ds_raw_dirs: List[Path] = []
        if raw_main.exists() and raw_main.is_dir():
            ds_raw_dirs.append(raw_main)
        if raw_anom.exists() and raw_anom.is_dir():
            ds_raw_dirs.append(raw_anom)

        out.append(DatasetPaths(dataset=ds, features_dir=features_dir, raw_dirs=ds_raw_dirs, results_dir=results_dir))

    return out


def step_extract_features(ds: DatasetPaths) -> None:
    ds.features_dir.mkdir(parents=True, exist_ok=True)
    out_path = ds.features_dir / "features.npy"
    if out_path.exists():
        print(f"\n[跳过] 已存在：{out_path}")
        return

    if not ds.raw_dirs:
        print(f"\n[提示] 未找到 {ds.dataset} 的 raw 目录（期望在 {RAW_ROOT_DEFAULT} 下）。")
        manual = input("请手动输入一个或多个图片目录（用分号 ; 分隔），或直接回车跳过提取：").strip().strip('"')
        if not manual:
            return
        dirs = [Path(p.strip()).resolve() for p in manual.split(";") if p.strip()]
    else:
        print(f"\n将从以下目录提取图像特征：")
        for d in ds.raw_dirs:
            print(f" - {d}")
        if not ask_yes_no("是否按上述目录执行特征提取？", default_yes=True):
            manual = input("请手动输入一个或多个图片目录（用分号 ; 分隔），或直接回车跳过提取：").strip().strip('"')
            if not manual:
                return
            dirs = [Path(p.strip()).resolve() for p in manual.split(";") if p.strip()]
        else:
            dirs = ds.raw_dirs

    if ask_yes_no("是否使用【特征提取】默认参数（resnet18 / auto / batch=64 / 不递归）？", default_yes=True):
        model = "resnet18"
        device = "auto"
        batch_size = "64"
        recursive = False
    else:
        model = choose_from_list("选择 backbone：", ["resnet18", "resnet50"], allow_multi=False, default_idxs=[0])[0]
        device = choose_from_list("选择 device：", ["auto", "cpu", "cuda"], allow_multi=False, default_idxs=[0])[0]
        batch = input("batch-size（默认 64）：").strip()
        batch_size = batch if batch else "64"
        recursive = ask_yes_no("是否递归扫描子目录（--recursive）？", default_yes=False)

    args = [
        "--dirs",
        *[str(d) for d in dirs],
        "--out-dir",
        str(ds.features_dir),
        "--model",
        model,
        "--device",
        device,
        "--batch-size",
        str(batch_size),
    ]
    if recursive:
        args.append("--recursive")

    print("\n[执行] 提取高维特征 -> features.npy")
    run_py(CODE_DIR / "extract_image_features.py", args)


def step_dimred(ds: DatasetPaths, *, methods: List[str]) -> List[str]:
    """确保生成低维 <METHOD>_features.npy；返回实际可用的方法列表（存在/生成成功）。"""
    high_path = ensure_features_file(ds.features_dir, ds.dataset)

    available: List[str] = []
    for m in methods:
        out_path = ds.features_dir / f"{m}_features.npy"
        if out_path.exists() or (ds.features_dir / f"{m}_embedding.npy").exists():
            print(f"\n[跳过] 已存在低维文件：{m}（{out_path.name} 或 {m}_embedding.npy）")
            available.append(m)
            continue

        print(f"\n[执行] 降维 {m} -> {m}_features.npy")
        args = ["--features", str(high_path), "--method", m.lower(), "--out-dir", str(ds.features_dir)]
        try:
            run_py(CODE_DIR / "compute_embeddings.py", args)
        except subprocess.CalledProcessError as e:
            print(f"[失败] 计算 {m} 失败：{e}")
            if ask_yes_no("是否继续其它方法？", default_yes=True):
                continue
            raise
        available.append(m)

    return available


def step_boundary(ds: DatasetPaths, *, detectors: List[str], methods: List[str]) -> None:
    high_path = ensure_features_file(ds.features_dir, ds.dataset)

    # 主流程默认：减少交互；参数尽量走 detector 自身默认（尤其 knn_distance 的 KNN_DISTANCE_K/threshold_ratio）。
    # 如需覆盖 k/d/threshold 等，请直接运行 detect_boundaries.py 并显式传参。
    norm_high = "zscore"
    norm_low = "minmax"

    for det in detectors:
        print(f"\n=== 边界检测器：{det} ===")

        # 1) 高维只跑一次
        args_high = [
            "--high",
            str(high_path),
            "--detector",
            det,
            "--normalize-high",
            norm_high,
            "--normalize-low",
            norm_low,
            "--out-dir",
            str(ds.features_dir),
            "--skip-low",
        ]
        run_py(CODE_DIR / "detect_boundaries.py", args_high)

        # 2) 低维：对每个方法各跑一次（只跑 low，避免重复高维）
        for m in methods:
            low_path = resolve_low_features_path(ds.features_dir, m)
            args_low = [
                "--high",
                str(high_path),
                "--low",
                str(low_path),
                "--detector",
                det,
                "--normalize-high",
                norm_high,
                "--normalize-low",
                norm_low,
                "--out-dir",
                str(ds.features_dir),
                "--skip-high",
            ]
            run_py(CODE_DIR / "detect_boundaries.py", args_low)


def step_mapping_and_html(ds: DatasetPaths, *, methods: List[str], detectors: List[str]) -> None:
    files_list = ds.features_dir / "files.txt"
    has_files_list = files_list.exists()

    for m in methods:
        print(f"\n=== Mapping：{m} ===")

        mapping_args = [
            "--features-dir",
            str(ds.features_dir),
            "--prefix",
            m,
            "--results-dir",
            str(ds.results_dir),
        ]
        if has_files_list:
            mapping_args += ["--files-list", str(files_list)]

        run_py(CODE_DIR / "make_embedding_mapping.py", mapping_args)

        for det in detectors:
            print(f"\n=== HTML：{m}_{det} ===")
            html_args = [
                "--prefix",
                m,
                "--detector",
                det,
                "--features-dir",
                str(ds.features_dir),
                "--results-dir",
                str(ds.results_dir),
                "--img-base",
                "http://localhost:5678",
            ]
            run_py(CODE_DIR / "boundary_analysis.py", html_args)


def main():
    parser = argparse.ArgumentParser(description="交互式入口：跑通完整流程（特征提取→边界→降维→边界→mapping→HTML）")
    parser.add_argument("--non-interactive", action="store_true", help="保留给后续扩展：当前仍以交互为主")
    args = parser.parse_args()

    if args.non_interactive:
        print("当前版本 main.py 以交互为主；--non-interactive 暂未实现具体参数化。")

    print("\n=== DR-Visualization 全流程入口 ===")
    datasets = pick_datasets_interactive()

    dimred_methods_all = get_dimred_methods()
    detectors_all = get_detectors()

    dimred_available, dimred_unavailable = filter_dimred_by_deps(dimred_methods_all)
    detectors_available, detectors_unavailable = filter_detectors_by_deps(detectors_all)

    if not dimred_available:
        raise RuntimeError("当前环境没有任何可用的降维方法（至少应有 PCA）。请检查依赖与代码安装是否完整。")
    if not detectors_available:
        raise RuntimeError("当前环境没有任何可用的边界检测方法（可能缺少 scipy / scikit-learn）。")

    if dimred_unavailable:
        print("\n[依赖提示] 以下降维方法当前不可用：")
        for m, reason in dimred_unavailable:
            print(f" - {m}: {reason}")
    if detectors_unavailable:
        print("\n[依赖提示] 以下检测方法当前不可用：")
        for d, reason in detectors_unavailable:
            print(f" - {d}: {reason}")

    # 选择降维方法（可多选）。
    chosen_dimred = choose_from_list(
        "选择降维方法（将生成 <METHOD>_features.npy；可多选）：",
        dimred_available,
        allow_multi=True,
        default_idxs=[0],
    )

    chosen_detectors = choose_from_list(
        "选择边界检测方法（可多选）：",
        detectors_available,
        allow_multi=True,
        default_idxs=[0],
    )

    # 选择是否跑哪些阶段
    fe_ok, fe_reason = feature_extraction_available()
    if not fe_ok:
        print(f"\n[依赖提示] 特征提取不可用：{fe_reason}")
        do_extract = False
    else:
        do_extract = ask_yes_no("是否执行【特征提取】步骤（raw -> features.npy）？", default_yes=True)
    do_dimred = ask_yes_no("是否执行【降维】步骤（features.npy -> <METHOD>_features.npy）？", default_yes=True)
    do_boundary = ask_yes_no("是否执行【边界检测】步骤（高维+低维）？", default_yes=True)
    do_mapping = ask_yes_no("是否执行【mapping+HTML】步骤？", default_yes=True)

    for ds in datasets:
        print(f"\n\n############################")
        print(f"数据集：{ds.dataset}")
        print(f"features 目录：{ds.features_dir}")
        print(f"results 目录：{ds.results_dir}")
        print("############################")

        # 给用户一个“当前已有方法”的提示
        if ds.features_dir.exists():
            existing = find_existing_methods(ds.features_dir)
            if existing:
                print(f"已检测到该数据集已有低维方法文件：{existing}")

        try:
            if do_extract:
                step_extract_features(ds)

            # 确保 features.npy（或兼容文件）存在
            _ = ensure_features_file(ds.features_dir, ds.dataset)

            methods_available = chosen_dimred
            if do_dimred:
                methods_available = step_dimred(ds, methods=chosen_dimred)
            else:
                # 不做降维也要保证低维文件存在，否则后面 mapping/html 会失败
                for m in chosen_dimred:
                    _ = resolve_low_features_path(ds.features_dir, m)

            if do_boundary:
                step_boundary(ds, detectors=chosen_detectors, methods=methods_available)

            if do_mapping:
                ds.results_dir.mkdir(parents=True, exist_ok=True)
                step_mapping_and_html(ds, methods=methods_available, detectors=chosen_detectors)

            print(f"\n[完成] {ds.dataset} 流程结束。")

        except Exception as e:
            print(f"\n[错误] 数据集 {ds.dataset} 处理失败：{e}")
            if not ask_yes_no("是否继续处理下一个数据集？", default_yes=True):
                raise

    print("\n全部数据集处理完成。")
    print("\n如需看图：请先运行 `py code/image_server.py`，再打开 results/<dataset>-results/<METHOD>.html")


if __name__ == "__main__":
    main()
