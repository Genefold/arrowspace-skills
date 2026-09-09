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

### Two pipelines: EigenMaps vs EnergyMaps

| Pipeline | Build call | Search calls | Construction |
|---|---|---|---|
| EigenMaps | `build` / `build_full` / `build_and_store` | `search`, `search_hybrid`, `search_batch`, `search_linear_sorted` | Cosine kNN graph + λ-graph Laplacian |
| EnergyMaps | `build_energy` | `search_energy`, `search_linear_sorted` | Energy-distance kNN — **no cosine in construction or search** |

### EnergyMaps pipeline

Build with `build_energy(items, energy_params, graph_params)`. Both dicts are required positional arguments; every `energy_params` key is optional (defaults shown):

```python
energy_params = {
    "optical_tokens": 50,   # target centroids after optical compression; None disables compression
    "trim_quantile": 0.1,   # fraction of high-norm items trimmed per spatial bin
    "eta": 0.1,             # diffusion step size for heat-flow smoothing over L0
    "steps": 4,             # number of diffusion iterations
    "split_quantile": 0.9,  # quantile threshold for splitting high-dispersion centroids
    "neighbor_k": 20,       # neighbourhood size for dispersion / local statistics
    "split_tau": 0.15,      # offset magnitude when splitting centroids along the local gradient
    "w_lambda": 1.0,        # weight of the lambda-proximity term in the energy distance
    "w_disp": 0.5,          # weight of the dispersion-difference term
    "w_dirichlet": 0.25,    # weight of the Rayleigh-Dirichlet term over feature deltas
    "candidate_m": 32,      # candidates evaluated before selecting the k nearest (M >= k)
}
aspace, gl = ArrowSpaceBuilder().build_energy(items, energy_params, graph_params)
```

Edge weights combine three terms weighted by `w_lambda` / `w_disp` / `w_dirichlet`: the lambda delta (spectral proximity), the dispersion (graph) delta, and a Dirichlet term over feature deltas. `optical_tokens` compresses the corpus into centroids by spatial binning with low-activation pooling; diffusion (`eta`, `steps`) smooths the compressed graph and high-dispersion centroids are split into sub-centroids (`split_quantile`, `split_tau`, `neighbor_k`). Query with `aspace.search_energy(q, gl, k)`; `search_batch` raises `NotImplementedError` for energy indexes — use single-row `search`, `search_hybrid`, or `search_linear_sorted` instead.

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

### Search variants

| Call | Behaviour |
|---|---|
| `search(q, gl, tau, k=None)` | Blended cosine + lambda-tau top-k; `k` overrides the build-time `topk` |
| `search_batch(Q, gl, tau, k=None)` | Batch search; degenerate rows (lambda ~0) yield `None` in that slot instead of aborting the batch. Not available for energy indexes (`NotImplementedError`) |
| `search_hybrid(q, gl, tau)` | Blended scoring plus pure-cosine candidates |
| `search_energy(q, gl, k)` | EnergyMaps indexes only (see `build_energy`) |
| `search_linear_sorted(q, gl, k)` | Sorted-lambda scan; works with both pipelines |

Contract: `search`, `search_hybrid`, and the batch path raise `ValueError` (not `PanicException`) on degenerate (lambda ~0), non-finite, or dimension-mismatched queries. `search_batch` is per-row failure tolerant — non-finite or mismatched batches still raise, since those affect every row.

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

### Sequencing and motif/subgraph analysis

Order items along the graph spectrum, or spot motifs and subgraphs on the Laplacian:

```python
from arrowspace import sequence_by_lambda, sequence_by_graph

seq = sequence_by_lambda(aspace.lambdas())   # spectral curriculum, ascending
seq.order        # node indices in sequence order
seq.positions    # per-step coordinate: lambda score (sequence_by_lambda),
                 # DFS discovery depth within its component (sequence_by_graph)
seq.components   # connected components traversed

seriated = sequence_by_graph(gl)   # MST-chain seriation of the Laplacian
```

Motif and subgraph spotting (`cfg` dicts optional; defaults apply per key):

```python
# EigenMaps builds: motifs on the F x F bootstrap Laplacian — returned ids
# are FEATURE dimensions, not items (node-space contract)
motifs = aspace.spot_motives_eigen(gl, {"top_l": 16, "min_triangles": 2})

# EigenMaps builds: item-space motifs on the centroid graph; every item must
# carry a cluster assignment (outlier-free) and n_clusters >= 2;
# ids are item indices in 0..nitems
motifs = aspace.spot_motives_eigen_items(gl, {"top_l": 16, "min_triangles": 2})

# EnergyMaps builds: motifs on the subcentroid graph, returned as item indices
motifs = aspace.spot_motives_energy(gl, {"top_l": 16, "min_triangles": 2})

# EnergyMaps builds: motif-anchored subgraphs — dicts with node_indices, item_indices,
# rayleigh (computed only when cfg sets rayleigh_max), nnodes, nfeatures
subgs = aspace.spot_subg_motives(gl, {"min_size": 4, "rayleigh_max": None})

# Centroid hierarchy — dicts with level, node_indices, root_indices, nnodes, nfeatures;
# cfg accepts graph params plus hierarchy keys min_centroids (8) and max_depth (2)
levels = aspace.spot_subg_centroids(gl, {"max_depth": 2, "min_centroids": 2})
```

Mode requirements: `spot_motives_eigen` / `spot_motives_eigen_items` need an EigenMaps build; `spot_motives_energy` / `spot_subg_motives` need an EnergyMaps build (`build_energy`) and raise `ValueError` otherwise.

Config keys (defaults): motif cfg `top_l=16` (prune to top-L strongest neighbours), `min_triangles=2`, `min_clust=0.4`, `max_motif_size=32`, `max_sets=256`, `jaccard_dedup=0.8`; subgraph cfg `min_size=3`, `rayleigh_max=None`; centroid-hierarchy cfg additionally accepts `min_centroids=8`, `max_depth=2`, and the graph-param keys — unknown keys are rejected.

## Rust API surface (arrowspace-rs 0.28)

For Rust users ([docs.rs/arrowspace](https://docs.rs/arrowspace)); the Python bindings wrap the same core:

- Builder defaults match the 0.28 defaults above (`eps=0.5`, `k=12`, `topk=6`, `p=2.0`, `sigma=None` meaning σ := eps); override via `with_lambda_graph(eps, k, topk, p, sigma_override)`.
- `TauMode` synthesis policies for the per-item lambda-tau score (distinct from the query-time alpha-beta blend `tau`): `Median` (default), `Mean`, `Percentile(f64)`, `Fixed(f64)`.
- Fallible variants `try_prepare_query_item` / `try_search_lambda_aware` return `ArrowSpaceError::DegenerateLambda` instead of panicking; the panicking twins `prepare_query_item` / `search_lambda_aware` are deprecated since 0.27.
- `range_search` for radius queries.
- `add_items` / `mul_items` / `scale_item` mutate items and recompute lambdas.
- `build_for_persistence(data, PipelineKind)` — `PipelineKind` is `Eigen` or `Energy(EnergyParams)` and parses from `"eigen"` / `"energy"` via `FromStr`.
- `from_config` reconstructs an index from a typed config map.
- `degenerate_lambda_count()` — Rust-side quality gate mirroring the lambda check above.
- Sequencing and motif analysis are available Rust-side under the `analysis` modules (`sequencing`, `motives`, `subgraphs`).

## Resources

- [JOSS paper](https://doi.org/10.21105/joss.09002)
- [Design article: Semantic Basins](https://www.tuned.org.uk/posts/020_arrowspace_semantic_basins_part2)
- [Presentation: Spectral Indexing](https://docs.google.com/presentation/d/1f1Zu3FTXltbsXLonflG-7yra4l383B6sbhIgnbx226g/)
- [Presentation: Deep-dive](https://docs.google.com/presentation/d/1Mtz-_85qpVROnp4U2VrnlSHn0266Z1yc_HfjUtfxYLs)
- [Complex examples: pyarrowspace test suite](https://github.com/tuned-org-uk/pyarrowspace/tree/main/tests)
- [Hyperparameter tuning: arrowspace_tuner](https://pypi.org/project/arrowspace-tuner/)
- [Rust source](https://github.com/Mec-iS/arrowspace-rs)
- [Rust API docs](https://docs.rs/arrowspace)
- [Python source](https://github.com/tuned-org-uk/pyarrowspace)

## See also

`skills/arrowspace-core.md`, `skills/arrowspace-search.md`, `skills/arrowspace-spectral.md` for focused skill definitions.
