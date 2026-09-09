from __future__ import annotations

import numpy as np


def tune_tau(
    aspace,
    gl,
    queries: np.ndarray,
    ground_truth: list[list[int]],
    tau_range: list[float] | None = None,
) -> float:
    """
    Simple tau grid search over recall@k.

    ``tau`` is the alpha-beta blend weight between cosine similarity and
    the spectral lambda-tau score: ``tau = 1.0`` is pure cosine ranking
    and lower values blend in spectral coherence.

    Parameters
    ----------
    aspace : ArrowSpace instance
    gl : graph laplacian from build()
    queries : np.ndarray, shape (M, D)
    ground_truth : list of list of int
        Relevant item indices per query.
    tau_range : list of float, optional
        Defaults to [0.0, 0.25, 0.5, 0.75, 1.0]. Values outside [0, 1]
        extrapolate the blend and are not recommended.

    Returns
    -------
    float : tau value with highest mean recall@k.
    """
    if tau_range is None:
        tau_range = [0.0, 0.25, 0.5, 0.75, 1.0]

    if not any(len(r) > 0 for r in ground_truth):
        raise ValueError("No non-empty ground truth entries provided")

    best_tau = tau_range[0]
    best_recall = 0.0

    for tau in tau_range:
        recalls = []
        for q, relevant in zip(queries, ground_truth):
            hits = aspace.search(q, gl, tau=tau)
            retrieved = {h[0] for h in hits}
            relevant_set = set(relevant)
            if relevant_set:
                recall = len(retrieved & relevant_set) / len(relevant_set)
                recalls.append(recall)
        mean_recall = float(np.mean(recalls)) if recalls else 0.0
        if mean_recall > best_recall:
            best_recall = mean_recall
            best_tau = tau

    return best_tau


def search_with_recall(
    aspace,
    gl,
    query: np.ndarray,
    tau: float = 1.0,
    max_results: int = 10,
) -> list[tuple[int, float]]:
    """
    Search and return top hits with their λτ scores.

    Parameters
    ----------
    aspace : ArrowSpace instance
    gl : graph laplacian
    query : np.ndarray, shape (D,)
    tau : float, alpha-beta blend weight (1.0 = pure cosine; lower adds
        spectral lambda-tau weighting)
    max_results : int, maximum number of results to return
        The actual count is bounded by the index's build-time ``topk``;
        this helper only truncates the returned hits.

    Returns
    -------
    list of (index, score) tuples.
    """
    all_hits = aspace.search(query, gl, tau=tau)
    return all_hits[:max_results]
