from __future__ import annotations

import numpy as np


def explain_spectral_properties(gl) -> dict:
    """
    Extract interpretable spectral diagnostics from a graph laplacian.

    Parameters
    ----------
    gl : graph laplacian object from ArrowSpaceBuilder.build()

    Returns
    -------
    dict with keys: n_nodes, eigvals_range, fiedler_value, spectral_gap,
    condition_number_estimate.
    """
    eigvals = np.sort(getattr(gl, "eigvals", getattr(gl, "lambdas", None)))
    if eigvals is None:
        return {"error": "no eigenvalues found in graph laplacian object"}

    fiedler = float(eigvals[1]) if len(eigvals) > 1 else float(eigvals[0])
    spectral_gap = float(eigvals[1] - eigvals[0]) if len(eigvals) > 1 else 0.0
    cond_est = float(eigvals[-1] / eigvals[0]) if eigvals[0] > 1e-12 else float("inf")

    return {
        "n_nodes": len(eigvals),
        "eigval_min": float(eigvals[0]),
        "eigval_max": float(eigvals[-1]),
        "fiedler_value": fiedler,
        "spectral_gap": spectral_gap,
        "condition_number_estimate": cond_est,
    }


def spectral_summary(gl) -> str:
    """
    Human-readable summary of the spectral properties of a graph laplacian.
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
