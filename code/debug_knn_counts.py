"""Quick diagnostics for knn_distance: boundary counts vs k.

Run:
  python code/debug_knn_counts.py --dataset yalefaces --method PCA

It mirrors detect_boundaries.py default normalizations:
- high: zscore
- low:  minmax
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def zscore(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std


def minmax(x: np.ndarray) -> np.ndarray:
    min_vals = x.min(axis=0)
    max_vals = x.max(axis=0)
    denom = max_vals - min_vals
    denom[denom == 0] = 1.0
    return (x - min_vals) / denom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="yalefaces")
    ap.add_argument("--method", default="PCA")
    ap.add_argument("--threshold-ratio", type=float, default=0.7)
    ap.add_argument("--threshold-mode", type=str, default="ratio")
    ap.add_argument(
        "--ks",
        type=str,
        default="5,10,15,30,50,100,150,200",
        help="Comma-separated k list",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    features_dir = repo_root / "data" / f"{args.dataset}-features"
    high_path = features_dir / "features.npy"
    low_path = features_dir / f"{args.method.upper()}_features.npy"

    if not high_path.exists():
        raise SystemExit(f"missing: {high_path}")
    if not low_path.exists():
        raise SystemExit(f"missing: {low_path}")

    import sys

    sys.path.insert(0, str((repo_root / "code").resolve()))
    from detection.knn_distance import detect_knn_distance  # type: ignore

    xh_raw = np.load(str(high_path))
    xl_raw = np.load(str(low_path))

    xh = zscore(xh_raw)
    xl = minmax(xl_raw)

    ks = [int(s.strip()) for s in str(args.ks).split(",") if s.strip()]

    print(f"dataset={args.dataset} N={xh.shape[0]} highD={xh.shape[1]} lowD={xl.shape[1]}")
    print(f"threshold={args.threshold_mode}@{args.threshold_ratio}")
    print("k\thigh_boundary\tlow_boundary")
    for k in ks:
        rh = detect_knn_distance(xh, k=k, threshold_mode=args.threshold_mode, threshold_ratio=args.threshold_ratio)
        rl = detect_knn_distance(xl, k=k, threshold_mode=args.threshold_mode, threshold_ratio=args.threshold_ratio)
        print(f"{k}\t{len(rh.boundary_indices)}\t\t{len(rl.boundary_indices)}")


if __name__ == "__main__":
    main()
