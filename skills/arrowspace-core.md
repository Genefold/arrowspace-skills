# ArrowSpace Core

Building an ArrowSpace index from vector embeddings.

## When to use

You have a set of items represented as dense vectors and want to build a spectral search index.

## How it works

ArrowSpace constructs a k-NN graph from the item vectors, computes the graph Laplacian, and extracts its eigenstructure. The resulting signal graph encodes both proximity and structural role. After building, each item has a $$λτ$$ (lambda-tau) score reflecting its spectral coherence.

## Steps

1. Prepare items as `np.ndarray` with dtype `float64`, shape `(N, D)`.
2. Choose graph parameters: `eps` (radius), `k` (neighbours), `sigma` (RBF width).
3. Call `ArrowSpaceBuilder().build(params, items)`.
4. Store the returned `(aspace, gl)` pair.

## The $$λτ$$ scores

After building, the ArrowSpace instance exposes per-item spectral scores:

```python
# λτ scores by item index: result[i] = score for the i-th item
scores = aspace.lambdas()

# Sorted ascending: list of (score, item_index) tuples
ranked = aspace.lambdas_sorted()
```

- **`lambdas()`** returns an array indexed by insertion order (item 0, item 1, ...). Higher score = more spectrally coherent.
- **`lambdas_sorted()`** returns `(score, index)` pairs sorted by score, from least to most coherent.

The $$λτ$$ score is distinct from graph eigenvalues — it is a per-item blend of Rayleigh quotient and Laplacian dispersion that characterises each item's structural role.

## Key parameters

- `eps`: neighbourhood radius. Smaller values produce sparser graphs.
- `k`: number of nearest neighbours. Should be > log(N).
- `sigma`: RBF kernel width. Controls how quickly affinity decays with distance.
- `topk`: number of results returned by search (can be overridden at query time).

## Default heuristic

```python
k = min(max(3, N // 10), 64)
eps = 1.0 if D <= 128 else 2.0
```

## References

- JOSS paper: https://doi.org/10.21105/joss.09002
- API docs: https://github.com/tuned-org-uk/pyarrowspace
