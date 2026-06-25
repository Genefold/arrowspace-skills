from __future__ import annotations

import numpy as np
from arrowspace import ArrowSpaceBuilder


def suggest_params(n_items: int, dim: int) -> dict:
    """
    Heuristic defaults for ArrowSpace graph parameters.

    Parameters
    ----------
    n_items : int
        Number of items in the dataset.
    dim : int
        Dimensionality of the embedding space.

    Returns
    -------
    dict with keys eps, k, topk, p, sigma.
    """
    k = min(max(3, int(n_items / 10)), 64)
    eps = 1.0 if dim <= 128 else 2.0
    return {
        "eps": eps,
        "k": k,
        "topk": min(10, n_items),
        "p": 2.0,
        "sigma": 1.0,
    }


def build_index(
    items: np.ndarray,
    params: dict | None = None,
) -> tuple:
    """
    Build an ArrowSpace index from a dense embedding array.

    Parameters
    ----------
    items : np.ndarray, shape (N, D), dtype float64
    params : dict, optional
        Graph parameters. Auto-suggested if None.

    Returns
    -------
    (aspace, gl) tuple from ArrowSpaceBuilder.build().
    """
    if params is None:
        params = suggest_params(items.shape[0], items.shape[1])
    return ArrowSpaceBuilder().build(params, items)
