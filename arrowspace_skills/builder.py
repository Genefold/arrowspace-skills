from __future__ import annotations

import numpy as np
from arrowspace import ArrowSpaceBuilder


def suggest_params(n_items: int, dim: int) -> dict:
    """
    Heuristic defaults for ArrowSpace graph parameters.

    Aligned with the arrowspace 0.28 builder defaults and the documented
    0.5-4.0 eps regime:
    - eps starts at 0.5 for general use, raised with dimensionality
    - k scales with dataset size, clamped to [12, 25]
    - topk = 6 (0.28 default retrieval count)
    - p = 2.0 (quadratic decay, default)
    - sigma = None (defaults to eps)

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
    k = min(max(12, int(n_items / 50)), 25)
    # Higher dims spread cosine distances; raise eps to maintain connectivity
    eps = 0.5 if dim <= 128 else 1.0 if dim <= 768 else 2.0
    return {
        "eps": eps,
        "k": k,
        "topk": 6,
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

    Raises
    ------
    ValueError
        If items is not a 2-D float64 array.
    """
    if items.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {items.shape}")
    items = np.ascontiguousarray(items, dtype=np.float64)
    if params is None:
        params = suggest_params(items.shape[0], items.shape[1])
    return ArrowSpaceBuilder().build(params, items)
