# ArrowSpace Spectral Analysis

Understanding the spectral properties of your vector dataset.

## When to use

- You want to characterise the structure of an embedding space.
- You need to detect whether a dataset has meaningful spectral separation.
- You are debugging a poorly performing ArrowSpace index.

## Key spectral diagnostics

### Fiedler value

The second smallest eigenvalue of the graph Laplacian. Low Fiedler value (< 0.1) indicates a poorly connected graph — items form isolated clusters or the k-NN graph is too sparse.

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

## References

- JOSS paper §2: Graph Laplacian and spectral methods
- Design article: https://www.tuned.org.uk/posts/020_arrowspace_semantic_basins_part2
- Deep-dive presentation: https://docs.google.com/presentation/d/1Mtz-_85qpVROnp4U2VrnlSHn0266Z1yc_HfjUtfxYLs
