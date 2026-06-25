from __future__ import annotations

import numpy as np
from arrowspace import ArrowSpaceBuilder


def suggest_params(n_items: int, dim: int) -> dict:
    """
    Heuristic defaults for ArrowSpace graph parameters.

    Follows the recommendations from pyarrowspace GRAPH_VARIABLES.md:
    - eps starts at 0.1 for general use, adjusted by dimensionality
    - k scales with dataset size, clamped to [3, 25]
    - p = 2.0 (quadratic decay, default)
    - sigma = None (aligns knee to eps)

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
    k = min(max(3, int(n_items / 50)), 25)
    topk = 3 if k <= 5 else 4
    # Higher dims may need larger eps to maintain connectivity
    eps = 0.1 if dim <= 128 else 0.2 if dim <= 768 else 0.5
    return {
        "eps": eps,
        "k": k,
        "topk": topk,
        "p": 2.0,
        "sigma": None,
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
