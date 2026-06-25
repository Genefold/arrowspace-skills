from __future__ import annotations

import numpy as np


def explain_spectral_properties(gl) -> dict:
    """
    Extract interpretable spectral diagnostics from a GraphLaplacian.

    Computes the eigendecomposition of the dense Laplacian matrix.
    For large graphs (N > 5000), consider using sparse eigensolvers.

    Parameters
    ----------
    gl : GraphLaplacian from ArrowSpaceBuilder.build()

    Returns
    -------
    dict with keys: n_nodes, eigval_min, eigval_max, fiedler_value,
    spectral_gap, condition_number_estimate.
    """
    n = gl.nnodes
    dense = gl.to_dense()
    if isinstance(dense, np.ndarray) and dense.ndim == 2:
        eigvals = np.sort(np.linalg.eigvalsh(dense))
    else:
        return {"error": "expected dense matrix from gl.to_dense()"}

    fiedler = float(eigvals[1]) if len(eigvals) > 1 else float(eigvals[0])
    spectral_gap = float(eigvals[1] - eigvals[0]) if len(eigvals) > 1 else 0.0
    cond_est = float(eigvals[-1] / eigvals[0]) if eigvals[0] > 1e-12 else float("inf")

    return {
        "n_nodes": n,
        "eigval_min": float(eigvals[0]),
        "eigval_max": float(eigvals[-1]),
        "fiedler_value": fiedler,
        "spectral_gap": spectral_gap,
        "condition_number_estimate": cond_est,
    }


def spectral_summary(gl) -> str:
    """
    Human-readable summary of the spectral properties.
    """
    props = explain_spectral_properties(gl)
    if "error" in props:
        return props["error"]

    lines = [
        f"Nodes: {props['n_nodes']}",
        f"Eigenvalue range: [{props['eigval_min']:.4f}, {props['eigval_max']:.4f}]",
        f"Fiedler value: {props['fiedler_value']:.4f}",
        f"Spectral gap: {props['spectral_gap']:.4f}",
    ]
    if props["fiedler_value"] < 0.1:
        lines.append("Low Fiedler value — graph may be poorly connected.")
    if props["condition_number_estimate"] > 1000:
        lines.append("High condition number — consider increasing eps or k.")
    return "\n".join(lines)


def item_lambdas(aspace) -> np.ndarray:
    """
    Return per-item ``λτ`` scores from an ArrowSpace instance.

    The array is indexed by item insertion order:
    ``result[i]`` is the λτ score for the i-th item passed to
    ``ArrowSpaceBuilder.build()``.

    These scores blend Rayleigh quotient and Laplacian dispersion
    for each item, reflecting its structural role in the graph.
    Higher score = more spectrally coherent.
    """
    return np.asarray(aspace.lambdas())


def sorted_lambdas(aspace) -> list[tuple[float, int]]:
    """
    Return ``(lambda, position)`` pairs sorted ascending by ``λτ`` score.

    Each element is ``(score, item_index)`` where ``item_index`` is the
    position of the item in the original array passed to the builder.
    The first element is the lowest-scoring item (least spectrally
    coherent); the last is the highest-scoring (most coherent).

    Useful for identifying spectral outliers and coherence ranking.
    """
    return list(aspace.lambdas_sorted())
