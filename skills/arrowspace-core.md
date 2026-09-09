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

Since 0.28 every parameter has a default (`eps=0.5`, `k=12`, `topk=6`, `p=2.0`, `sigma=eps`) and a partial dict is accepted. Defaults are a starting point — use `arrowspace_tuner` to compute parameters fitted to your dataset (label-free, Optuna-based):

```python
import arrowspace_tuner
from arrowspace import ArrowSpaceBuilder

graph_params = arrowspace_tuner.tune(items)   # discovers eps, k, topk, p, sigma
aspace, gl = ArrowSpaceBuilder().build(graph_params, items)
```

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

## Quality gate

A mistuned `eps` maps all $$λτ$$ scores near zero and produces an unusable index. Check after every build:

```python
import numpy as np
n_degenerate = int(np.sum(np.abs(aspace.lambdas()) < 1e-12))
```

Many zero lambdas mean: increase `eps`, then rebuild.

## Default heuristic

```python
k = min(max(12, N // 50), 25)
eps = 0.5 if D <= 128 else 1.0 if D <= 768 else 2.0
topk = 6
```

Matches `suggest_params()` from `arrowspace_skills` and the arrowspace 0.28 defaults. Note: this is a coarse heuristic — for corpus-fitted parameters use `arrowspace_tuner` (`arrowspace_tuner.tune(items)`), which discovers `eps`, `k`, `topk`, and `tau` automatically.

## References

- JOSS paper: https://doi.org/10.21105/joss.09002
- API docs: https://github.com/tuned-org-uk/pyarrowspace
- Complex examples: https://github.com/tuned-org-uk/pyarrowspace/tree/main/tests
- Tuner: https://pypi.org/project/arrowspace-tuner/
