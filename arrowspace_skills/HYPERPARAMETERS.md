# ArrowSpace Hyperparameters

Reference for the `graph_params` dictionary passed to `ArrowSpaceBuilder.build()`.

```python
from arrowspace import ArrowSpaceBuilder

params = {
    "eps": 0.5,         # distance cutoff
    "k": 12,            # max neighbours
    "topk": 6,          # results to retain
    "p": 2.0,           # kernel sharpness
    "sigma": None,      # scale (None → defaults to eps)
}
aspace, gl = ArrowSpaceBuilder().build(params, items)
```

These are the arrowspace 0.28 builder defaults; a partial dict is accepted. Defaults are a starting point — for parameters fitted to your dataset, see [Automated tuning with `arrowspace_tuner`](#automated-tuning-with-arrowspace_tuner) below.

## Parameter reference

### `eps` — distance cutoff

| Item | Value |
|---|---|
| **Type** | `float` |
| **Range** | $$[0, 2]$$ |
| **Default** | — (must be provided) |

**Description:** Maximum rectified cosine distance for edge inclusion:

$$d(u,v) = \sqrt{2(1 - \max(0, \cos(u,v)))}$$

Edges with $$d > \text{eps}$$ are discarded before the k-NN cap. This is the primary sparsity control.

**Effect of tuning:**

| Setting | Graph | When to use |
|---|---|---|
| Very small ($$< 0.1$$) | Very sparse; risk of disconnected components and zero $$λτ$$ scores | High-precision embeddings, tight clusters expected |
| Moderate ($$0.5$$–$$1.0$$) | Sparse but connected | General-purpose start point (0.28 default: 0.5) |
| Large ($$> 1.0$$) | Dense; many candidates pass threshold | High-dimensional or spread-out embeddings |

**Notes on scale:** ArrowSpace internally normalises items to unit norm, then computes cosine distances. If your raw embedding values are very small (e.g. $$10^{-2}$$), the dot products may be unstable at low `eps`. A common fix is to scale normalised embeddings before building (e.g. multiply by 12.0) and raise `eps` correspondingly (e.g. 1.0–1.5).

---

### `k` — max neighbours

| Item | Value |
|---|---|
| **Type** | `int` |
| **Range** | $$1 \dots N-1$$ |
| **Default** | — (must be provided) |

**Description:** Per-node cap on number of nearest neighbours retained after the `eps` threshold. The candidate edges are symmetrised and converted to a Laplacian.

**Effect of tuning:**

| Setting | Graph | When to use |
|---|---|---|
| Small (3–6) | Sparse, fast to build | Speed-critical, large N, or known clean manifold |
| Moderate (12–25) | Well-connected, stable spectra | General use (0.28 default: 12) |
| Large ($$> 25$$) | Dense, $$O(N \cdot k)$$ edges | Noisy embeddings, need robust connectivity |

Larger `k` increases memory and compute cost for both the item graph and the subsequent feature Laplacian.

---

### `p` — kernel exponent

| Item | Value |
|---|---|
| **Type** | `float` |
| **Range** | $$> 0$$ |
| **Default** | `2.0` |

**Description:** Exponent in the edge weight kernel:

$$w_{ij} = \frac{1}{1 + (d_{ij} / \sigma)^p}$$

Controls how sharply weights decay with distance.

**Effect of tuning:**

| Setting | Behaviour |
|---|---|
| $$p \approx 1$$ | Gentle, linear-ish decay; robust to distance variation |
| $$p = 2$$ | Quadratic decay (default) — balanced |
| $$p \geq 3$$ | Sharp, contrastive; selective near $$\sigma$$ but can amplify noise |

---

### `sigma` — kernel scale

| Item | Value |
|---|---|
| **Type** | `float` or `None` |
| **Range** | $$> 0$$ |
| **Default** | `None` → internally set to `eps` |

**Description:** Scale parameter in the weight kernel. When `None`, $$\sigma = \text{eps}$$ with a small floor, aligning the soft-decay knee to the distance threshold.

**Effect of tuning:**

| Setting | Behaviour |
|---|---|
| `None` | $$\sigma = \text{eps}$$ — knee aligns to cutoff; predictable across datasets |
| $$\sigma < \text{eps}$$ | Sharper kernel inside the allowed radius; more selective |
| $$\sigma > \text{eps}$$ | Flatter weights; the eps cutoff dominates edge selection |

---

### `topk` — retained results

| Item | Value |
|---|---|
| **Type** | `int` |
| **Range** | $$1 \dots N$$ |
| **Default** | `6` (0.28 default) |

**Description:** Number of results returned by `search` when no `k` override is passed. Any `k` passed at query time takes precedence, so this is not typically a tuning target.

---

## Quick reference

```python
# 0.28 defaults (general-purpose start)
{
    "eps": 0.5,
    "k": 12,
    "p": 2.0,
    "sigma": None,
}

# Better connectivity for larger corpora
{
    "eps": 0.5,
    "k": 25,
    "p": 2.0,
    "sigma": None,
}

# High-dimensional or spread-out embeddings (raise eps, more neighbours)
{
    "eps": 2.0,
    "k": 25,
    "p": 2.0,
    "sigma": None,
}
```

## Diagnosing problems

| Symptom | Likely cause | Fix |
|---|---|---|
| All $$\lambda\tau$$ scores are 0.0 | Graph disconnected — no edges formed | Increase `eps` or `k` |
| All $$\lambda\tau$$ scores near 1.0 | Graph too dense — everything is connected | Decrease `eps` or `k` |
| Search returns same results for all `tau` | Graph too dense | Reduce `k` |
| High condition number in Laplacian | Graph too sparse or disconnected | Increase `eps` or `k` |
| Poor recall on known neighbours | `eps` too restrictive | Increase `eps` or reduce `p` |
| Defaults underperform on your corpus | Corpus-specific structure | Use `arrowspace_tuner` (below) |

## Reference

- Authoritative source: [`GRAPH_VARIABLES.md`](https://github.com/tuned-org-uk/pyarrowspace/blob/main/GRAPH_VARIABLES.md) in pyarrowspace
- Complex examples: [pyarrowspace test suite](https://github.com/tuned-org-uk/pyarrowspace/tree/main/tests)
- Rust struct: [`GraphParams`](https://github.com/tuned-org-uk/arrowspace-rs/blob/main/src/graph.rs) in arrowspace-rs
- JOSS paper: https://doi.org/10.21105/joss.09002

## Automated tuning with `arrowspace_tuner`

Manual parameter search is tedious and corpus-dependent. The companion package [`arrowspace_tuner`](https://github.com/Genefold/arrowspace_tuner) uses Optuna to discover `eps`, `k`, `topk`, and `tau` automatically using a label-free spectral MRR proxy — a good alternative whenever defaults or heuristics are mentioned above.

```bash
pip install arrowspace-tuner
```

```python
import numpy as np
import arrowspace_tuner
from arrowspace import ArrowSpaceBuilder

embeddings = np.load("corpus.npy")  # shape (N, D) float64

# One-liner: discovers eps, k, topk (and tau for query time) — ~15 min on 50k corpus
graph_params = arrowspace_tuner.tune(embeddings)
aspace, gl = ArrowSpaceBuilder().build(graph_params, embeddings)
```

Or the power-user API with full control (returns the graph-params dict; the final build always uses the full corpus):

```python
from arrowspace_tuner import EpsTuner, ArrowSpaceBuilder

tuner = EpsTuner(
    n_trials=15,          # >= 10 recommended (pruning needs completed trials)
    sample_n=50_000,      # subsample for the study; final build uses all items
    eps_low=0.3,
    eps_high=4.0,
    k_low=3,
    k_high=40,
    n_probe=50,
    storage="sqlite:///tune.db",   # optional: resume interrupted runs
)
graph_params = tuner.fit(embeddings)
aspace, gl = ArrowSpaceBuilder().build(graph_params, embeddings)

print(graph_params)   # {"eps": ..., "k": ..., "topk": ..., "p": ..., "sigma": ...}
print(tuner.best_tau) # query-time blend weight, not part of graph_params
```

The objective blends retrieval coherence (spectral MRR proxy), graph connectivity (Fiedler value), and spectral richness — no ground-truth labels required. Install `arrowspace-tuner[report]` for CSV/HTML reporting (`tuner.save_report()`), and use multiple workers against the same SQLite storage for parallel runs.
