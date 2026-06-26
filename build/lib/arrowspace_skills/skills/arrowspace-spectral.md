# ArrowSpace Spectral Analysis

Understanding the spectral properties of your vector dataset and interpreting per-item $$λτ$$ scores.

## When to use

- You want to characterise the structure of an embedding space.
- You need to detect whether a dataset has meaningful spectral separation.
- You are debugging a poorly performing ArrowSpace index.
- You want to inspect per-item $$λτ$$ scores to find spectral outliers.

## Per-item $$λτ$$ scores

The ArrowSpace instance exposes two methods for accessing $$λτ$$ scores:

| Method | Returns | Description |
|---|---|---|
| `aspace.lambdas()` | `np.ndarray` shape `(N,)` | Scores indexed by item order — `result[i]` is the score for the i-th item passed to the builder |
| `aspace.lambdas_sorted()` | `list[(float, int)]` | `(score, item_index)` pairs sorted ascending by score |

These scores blend the Rayleigh quotient smoothness energy with Laplacian dispersion. They are **not** graph eigenvalues — they are per-item spectral signatures.

### Interpreting $$λτ$$ scores

- **High score** (~1.0): item occupies a spectrally coherent structural role. Good candidate for retrieval, stable under perturbation.
- **Low score** (~0.0): item is an outlier or sits at a structural boundary. May be noise or a bridging point between clusters.
- **Sudden drop** in the sorted list: indicates a spectral gap in the item set — a natural separation between clusters.

## Graph spectral diagnostics

These operate on the graph Laplacian matrix ($$L$$), not on the per-item $$λτ$$ scores.

### Fiedler value

The second smallest eigenvalue of $$L$$. Low Fiedler value (< 0.1) indicates a poorly connected graph — items form isolated clusters or the k-NN graph is too sparse.

### Spectral gap

The gap between the first and second eigenvalues. A large gap means a clear spectral separation between the strongest structural mode and the rest.

### Condition number estimate

Ratio of largest to smallest eigenvalue. A high condition number (> 1000) suggests numerical instability — increase `eps` or `k` to densify the graph.

## When to adjust parameters

| Symptom | Fix |
|---|---|
| Fiedler value < 0.1 | Increase `k` or `eps` |
| High condition number | Increase `eps` or `sigma` |
| All tau values give same results | Graph too dense, reduce `k` |
| Search returns very few hits | Reduce `tau` or increase `topk` |
| λτ scores all near 1.0 | Graph too dense — reduce `k` or `eps` |
| λτ scores all near 0.0 | Graph too sparse — increase `k` or `eps` |

## References

- JOSS paper §2: Graph Laplacian and spectral methods
- Design article: https://www.tuned.org.uk/posts/020_arrowspace_semantic_basins_part2
- Deep-dive presentation: https://docs.google.com/presentation/d/1Mtz-_85qpVROnp4U2VrnlSHn0266Z1yc_HfjUtfxYLs
