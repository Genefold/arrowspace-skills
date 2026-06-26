# ArrowSpace Hyperparameters

Reference for the `graph_params` dictionary passed to `ArrowSpaceBuilder.build()`.

```python
from arrowspace import ArrowSpaceBuilder

params = {
    "eps": 0.05,        # distance cutoff
    "k": 6,             # max neighbours
    "topk": 4,          # results to retain
    "p": 2.0,           # kernel sharpness
    "sigma": None,      # scale (None → defaults to eps)
}
aspace, gl = ArrowSpaceBuilder().build(params, items)
```

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
| Very small ($$< 0.01$$) | Very sparse; risk of disconnected components | High-precision embeddings, tight clusters expected |
| Moderate ($$0.05$$–$$0.15$$) | Sparse but connected | General-purpose start point |
| Large ($$> 1.0$$) | Dense; many candidates pass threshold | Low-precision or scaled embeddings; see *Notes on scale* below |

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
| Moderate (10–25) | Well-connected, stable spectra | General use |
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
| **Default** | Heuristic: `3` if `k ≤ 5`, `4` if `k < 10`, else `k` itself |

**Description:** Number of closest results retained per node during neighbour selection. Not typically a tuning target — it follows `k` by heuristic.

---

## Quick reference

```python
# Conservative start (sparse graph, stable spectral)
{
    "eps": 0.05,
    "k": 6,
    "p": 2.0,
    "sigma": None,
}

# Connected but sparse (good general default)
{
    "eps": 0.1,
    "k": 15,
    "p": 2.0,
    "sigma": None,
}

# Noisy or low-precision embeddings (scale up, raise eps)
{
    "eps": 1.2,
    "k": 25,
    "p": 1.5,
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

## Reference

- Authoritative source: [`GRAPH_VARIABLES.md`](https://github.com/tuned-org-uk/pyarrowspace/blob/main/GRAPH_VARIABLES.md) in pyarrowspace
- Rust struct: [`GraphParams`](https://github.com/tuned-org-uk/arrowspace-rs/blob/main/src/graph.rs) in arrowspace-rs
- JOSS paper: https://doi.org/10.21105/joss.09002

## Automated tuning with `arrowspace_tuner`

Manual parameter search is tedious and corpus-dependent. The companion package [`arrowspace_tuner`](https://github.com/Genefold/arrowspace_tuner) uses Optuna to discover optimal `eps`, `k`, and `tau` automatically using a label-free spectral MRR proxy.

```bash
pip install arrowspace-tuner
```

```python
import arrowspace_tuner
import numpy as np

embeddings = np.load("corpus.npy")  # shape (N, D) float64

# One-liner: discovers eps, k, tau in ~15 min on 50k corpus
aspace, gl = arrowspace_tuner.optuna(embeddings)

# Inspect the best params found
print(aspace, gl)

# Or use the power-user API with full control
from arrowspace_tuner import EpsTuner

tuner = EpsTuner(
    n_trials=15,
    eps_low=0.8,
    eps_high=10,
    k_low=15,
    k_high=40,
)
aspace, gl = tuner.fit(embeddings)
print(tuner.best_params)  # {"eps": 1.615, "k": 38, "tau": 0.114}
```

The objective blends retrieval coherence (spectral MRR proxy), graph connectivity (Fiedler value), and spectral richness — no ground-truth labels required.
