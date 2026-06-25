---
name: arrowspace
description: Spectral vector search using graph Laplacian eigenstructure. Build signal graphs, run λτ-indexed queries, and analyse spectral properties of vector datasets.
metadata:
  origin: Genefold AI
  version: 0.1.0
  package: arrowspace
---

# ArrowSpace

ArrowSpace is a vector database and search library that augments nearest-neighbour search with spectral graph features. It computes a Laplacian over the item graph and uses the Rayleigh quotient to produce a `λτ` (lambda-tau) score per item, enabling search that respects both semantic similarity and structural role.

## When to Activate

- You need vector similarity search that goes beyond cosine / L2
- Your dataset has latent structure that proximity metrics miss
- You want to characterise the spectral properties of an embedding space
- You need graph-based retrieval with spectral awareness

## Installation

```bash
pip install arrowspace
```

Or from source: see [pyarrowspace](https://github.com/tuned-org-uk/pyarrowspace).

## Core API

### Build an ArrowSpace index

```python
from arrowspace import ArrowSpaceBuilder
import numpy as np

items = np.array([[...], [...], ...], dtype=np.float64)
params = {"eps": 1.0, "k": 6, "topk": 3, "p": 2.0, "sigma": 1.0}
aspace, gl = ArrowSpaceBuilder().build(params, items)
```

`gl` is the graph laplacian containing eigenvalues, eigenvectors, and signal graph.

### Search

```python
query = np.array([...], dtype=np.float64)
hits = aspace.search(query, gl, tau=1.0)
# Returns list of (index, score) tuples
```

### Parameters

| Param | Default | Description |
|---|---|---|
| `eps` | 1.0 | Neighbourhood radius for graph construction |
| `k` | 6 | Number of nearest neighbours |
| `topk` | 3 | Number of top candidates to return |
| `p` | 2.0 | Distance norm (2 = Euclidean) |
| `sigma` | 1.0 | RBF kernel width |

### Tau (λτ) score

`tau` controls the spectral gate. Higher values include more items; lower values restrict to the most spectrally coherent candidates. Start at `tau=1.0` and tune based on recall-precision trade-off.

## Resources

- [JOSS paper](https://doi.org/10.21105/joss.09002)
- [Design article: Semantic Basins](https://www.tuned.org.uk/posts/020_arrowspace_semantic_basins_part2)
- [Presentation: Spectral Indexing](https://docs.google.com/presentation/d/1f1Zu3FTXltbsXLonflG-7yra4l383B6sbhIgnbx226g/)
- [Presentation: Deep-dive](https://docs.google.com/presentation/d/1Mtz-_85qpVROnp4U2VrnlSHn0266Z1yc_HfjUtfxYLs)
- [Rust source](https://github.com/Mec-iS/arrowspace-rs)
- [Python source](https://github.com/tuned-org-uk/pyarrowspace)

## See also

`skills/arrowspace-core.md`, `skills/arrowspace-search.md`, `skills/arrowspace-spectral.md` for focused skill definitions.
