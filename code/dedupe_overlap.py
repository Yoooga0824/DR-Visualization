"""
去重脚本：从正常集中剔除与异常集重复的图片（或文件），确保二者无交集。

通用性：只需指定数据集名（如 zero），脚本会默认在 data/raw 下使用：
- 正常目录：data/raw/<dataset>
- 异常目录：data/raw/Anomalous_<dataset>

支持两种判重方式：
- 按文件名（默认）：以文件名（不含路径，大小写不敏感）为键，速度快，适合常见场景。
- 按内容哈希：--by-hash 时启用，计算 MD5，能发现不同文件名但内容相同的文件。

安全策略：
- 默认 dry-run（试运行），仅打印将被删除的文件数量和样例，不做修改。
- 传 --apply 才会实际删除；或传 --move-to <dir> 将重复文件移动到该目录而不是删除。

用法（Windows PowerShell）：
  # 仅预览（zero vs Anomalous_zero）
  py code/dedupe_overlap.py --dataset zero

  # 按文件名删除（应用变更）
  py code/dedupe_overlap.py --dataset zero --apply

  # 按内容哈希判重，并移动到备份目录，而不是删除
  py code/dedupe_overlap.py --dataset zero --by-hash --move-to D:/backup/zero_dups
"""

import argparse
import hashlib
import os
from pathlib import Path
from typing import Iterable, Set, Tuple


def iter_files(root: Path, exclude: Path | None = None) -> Iterable[Path]:
    """递归遍历 root 下所有文件，若 exclude 在 root 子树中，则跳过 exclude 子树。"""
    if not root.exists():
        return []
    root = root.resolve()
    exclude = exclude.resolve() if exclude else None
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        if exclude and (exclude in p.parents or p == exclude):
            continue
        yield p


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open('rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_key_sets(anom_files: Iterable[Path], by_hash: bool) -> Tuple[Set[str], Set[str]]:
    """从异常集文件生成判重键集合。
    返回：(name_keys, hash_keys)
    - name_keys：文件名（不含路径，大小写不敏感）集合
    - hash_keys：MD5 哈希集合（仅当 by_hash=True 时会用到）
    """
    name_keys: Set[str] = set()
    hash_keys: Set[str] = set()
    for p in anom_files:
        name_keys.add(os.path.normcase(p.name))
        if by_hash:
            try:
                hash_keys.add(file_md5(p))
            except Exception:
                pass
    return name_keys, hash_keys


def plan_removals(normal_files: Iterable[Path], *, name_keys: Set[str], hash_keys: Set[str], by_hash: bool) -> list[Path]:
    to_remove: list[Path] = []
    for p in normal_files:
        name_key = os.path.normcase(p.name)
        if name_key in name_keys:
            to_remove.append(p)
            continue
        if by_hash:
            try:
                if file_md5(p) in hash_keys:
                    to_remove.append(p)
            except Exception:
                # 读取失败的文件，忽略
                pass
    return to_remove


def main():
    parser = argparse.ArgumentParser(description='从正常集剔除与异常集重复的文件（默认试运行）')
    parser.add_argument('--dataset', required=True, help='数据集名，例如 zero（正常目录为 data/raw/zero，异常目录为 data/raw/Anomalous_zero）')
    parser.add_argument('--raw-root', default=None, help='原始数据根目录，默认推断为 <repo>/data/raw')
    parser.add_argument('--by-hash', action='store_true', help='使用内容 MD5 判重（更严格，较慢）')
    parser.add_argument('--apply', action='store_true', help='实际执行删除（默认只打印不删除）')
    parser.add_argument('--move-to', default=None, help='将重复文件移动到该目录而不是删除（与 --apply 一起使用）')
    parser.add_argument('--limit', type=int, default=20, help='预览时最多展示多少条样例')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    raw_root = Path(args.raw_root) if args.raw_root else (repo_root / 'data' / 'raw')
    ds = args.dataset
    normal_dir = (raw_root / ds).resolve()
    anom_dir = (raw_root / f'Anomalous_{ds}').resolve()

    if not normal_dir.exists():
        raise SystemExit(f"正常目录不存在: {normal_dir}")
    if not anom_dir.exists():
        raise SystemExit(f"异常目录不存在: {anom_dir}")

    # 如果异常目录位于正常目录子树中，扫描正常目录时排除它
    anom_inside_normal = anom_dir in normal_dir.rglob('*') or (anom_dir in normal_dir.parents)
    normal_files = list(iter_files(normal_dir, exclude=anom_dir))
    anom_files = list(iter_files(anom_dir))

    print(f"正常目录: {normal_dir}")
    print(f"异常目录: {anom_dir}")
    print(f"正常文件数: {len(normal_files)}，异常文件数: {len(anom_files)}")

    name_keys, hash_keys = build_key_sets(anom_files, args.by_hash)
    to_remove = plan_removals(normal_files, name_keys=name_keys, hash_keys=hash_keys, by_hash=args.by_hash)

    print(f"将从正常集中剔除的文件数: {len(to_remove)}")
    for p in to_remove[: args.limit]:
        print(" -", p)
    if len(to_remove) > args.limit:
        print(f" ... 以及 {len(to_remove) - args.limit} 个更多")

    if not args.apply and not args.move_to:
        print("\n（试运行）未做任何改动。若要应用删除：追加 --apply；或使用 --move-to <目录> 进行移动备份。")
        return

    # 应用变更：删除或移动
    moved = 0
    removed = 0
    if args.move_to:
        target = Path(args.move_to).resolve()
        target.mkdir(parents=True, exist_ok=True)
        for src in to_remove:
            try:
                rel = src.relative_to(normal_dir)
            except ValueError:
                rel = Path(src.name)
            dst = (target / rel).resolve()
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                src.replace(dst)
                moved += 1
            except Exception as e:
                print(f"移动失败: {src} -> {dst} | {e}")
    else:
        # 真删除
        for src in to_remove:
            try:
                src.unlink(missing_ok=True)
                removed += 1
            except Exception as e:
                print(f"删除失败: {src} | {e}")

    print("\n操作完成：")
    if args.move_to:
        print(f"  已移动 {moved} 个文件到: {args.move_to}")
    else:
        print(f"  已删除 {removed} 个文件")


if __name__ == '__main__':
    main()
