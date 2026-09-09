---
name: arrowspace
description: Spectral vector search using graph Laplacian eigenstructure. Build signal graphs, run λτ-indexed queries, and analyse spectral properties of vector datasets.
metadata:
  origin: Genefold AI
  version: 0.28.1
  package: arrowspace
---

# ArrowSpace

ArrowSpace is a vector database and search library that augments nearest-neighbour search with spectral graph features. It computes a Laplacian over the item graph and uses the Rayleigh quotient to produce a $$λτ$$ (lambda-tau) score per item, enabling search that respects both semantic similarity and structural role.

## When to Activate

- You need vector similarity search that goes beyond cosine / L2
- Your dataset has latent structure that proximity metrics miss
- You want to characterise the spectral properties of an embedding space
- You need graph-based retrieval with spectral awareness

## Installation

```bash
pip install "arrowspace>=0.28"
```

Or from source: see [pyarrowspace](https://github.com/tuned-org-uk/pyarrowspace).

## Core API

### Build an ArrowSpace index

```python
from arrowspace import ArrowSpaceBuilder
import numpy as np

items = np.array([[...], [...], ...], dtype=np.float64)
params = {"eps": 0.5, "k": 12, "topk": 6, "p": 2.0, "sigma": None}
aspace, gl = ArrowSpaceBuilder().build(params, items)
```

Since 0.28 every key has a default (`eps=0.5`, `k=12`, `topk=6`, `p=2.0`, `sigma=eps`); a partial dict is accepted. Defaults are a starting point — to compute parameters fitted to your dataset, use [`arrowspace_tuner`](https://pypi.org/project/arrowspace-tuner/):

```python
import arrowspace_tuner
from arrowspace import ArrowSpaceBuilder

graph_params = arrowspace_tuner.tune(items)   # discovers eps, k, topk, p, sigma
aspace, gl = ArrowSpaceBuilder().build(graph_params, items)
```

`gl` is the graph Laplacian containing the Laplacian matrix (accessed via `gl.to_dense()` or `gl.to_csr()`).

### Per-item $$λτ$$ scores

After building, the ArrowSpace instance exposes spectral scores for every item:

```python
# Indexed by item insertion order: result[i] is score for i-th item
scores = aspace.lambdas()

# Sorted ascending: list of (score, item_index) tuples
ranked = aspace.lambdas_sorted()
```

- `lambdas()` — scores array aligned by item index (item 0, item 1, ...)
- `lambdas_sorted()` — `(score, index)` pairs sorted from least to most coherent

These are per-item spectral signatures, distinct from graph eigenvalues.

### Search

```python
query = np.array([...], dtype=np.float64)
hits = aspace.search(query, gl, tau=1.0)
# Returns list of (index, score) tuples; pass k=... to override topk
```

### Parameters

| Param | Default (0.28) | Description |
|---|---|---|
| `eps` | 0.5 | Neighbourhood radius for graph construction; raise for high-dimensional embeddings |
| `k` | 12 | Number of nearest neighbours |
| `topk` | 6 | Number of results returned by search (overridable per query via `k`) |
| `p` | 2.0 | Distance norm (2 = Euclidean) |
| `sigma` | = eps | RBF kernel width (`None` → eps) |

Note: instead of hand-picking `eps`/`k`/`tau`, `arrowspace_tuner` can discover them from your corpus (label-free, Optuna-based) — `pip install arrowspace-tuner`, then `arrowspace_tuner.tune(items)`.

### Tau ($$λτ$$ blend)

`tau` is the alpha-beta blend weight between cosine similarity and the spectral $$λτ$$ score: `tau` weights the cosine term, `1 - tau` the spectral term. `tau=1.0` is pure cosine ranking; lower values blend in structural coherence. Start at `tau=1.0` and lower it (e.g. 0.5–0.75) when retrieval should respect structural role. Values outside $$[0, 1]$$ extrapolate the blend and are not recommended.

After building, check for degenerate λτ scores — `np.sum(np.abs(aspace.lambdas()) < 1e-12)` should be ~0. Many zeros mean `eps` is too low: raise it and rebuild.

## Resources

- [JOSS paper](https://doi.org/10.21105/joss.09002)
- [Design article: Semantic Basins](https://www.tuned.org.uk/posts/020_arrowspace_semantic_basins_part2)
- [Presentation: Spectral Indexing](https://docs.google.com/presentation/d/1f1Zu3FTXltbsXLonflG-7yra4l383B6sbhIgnbx226g/)
- [Presentation: Deep-dive](https://docs.google.com/presentation/d/1Mtz-_85qpVROnp4U2VrnlSHn0266Z1yc_HfjUtfxYLs)
- [Complex examples: pyarrowspace test suite](https://github.com/tuned-org-uk/pyarrowspace/tree/main/tests)
- [Hyperparameter tuning: arrowspace_tuner](https://pypi.org/project/arrowspace-tuner/)
- [Rust source](https://github.com/Mec-iS/arrowspace-rs)
- [Python source](https://github.com/tuned-org-uk/pyarrowspace)

## See also

`skills/arrowspace-core.md`, `skills/arrowspace-search.md`, `skills/arrowspace-spectral.md` for focused skill definitions.
