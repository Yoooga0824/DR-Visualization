from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from .pca import pca_reduce
from .tsne import tsne_reduce
from .umap_ import umap_reduce


ReducerFn = Callable[..., np.ndarray]


_REDUCERS: Dict[str, ReducerFn] = {
    "pca": pca_reduce,
    "tsne": tsne_reduce,
    "umap": umap_reduce,
}


def list_methods() -> List[str]:
    return sorted(_REDUCERS.keys())


def get_reducer(name: str) -> ReducerFn:
    key = (name or "").strip().lower()
    if key not in _REDUCERS:
        raise KeyError(f"未知降维方法: {name}（可用：{list_methods()}）")
    return _REDUCERS[key]
