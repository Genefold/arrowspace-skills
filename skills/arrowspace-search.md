# ArrowSpace Vector Search

Querying an ArrowSpace index with $$λτ$$ scoring.

## When to use

You have a built ArrowSpace index and need to retrieve items similar to a query vector, ranked by spectral coherence.

## How it works

Each item is scored by a combination of its distance to the query and its spectral role in the graph (expressed by that item's $$λτ$$ score). The `tau` parameter controls the spectral gate: higher tau values include more items; lower tau values restrict to spectrally coherent candidates.

## Steps

1. Build an index (see `arrowspace-core.md`).
2. For a query vector `q`, call `aspace.search(q, gl, tau=tau)`.
3. The result is a list of `(index, score)` tuples sorted descending by score.

## Tuning tau

- Start at `tau = 1.0`.
- Lower tau (~0.1–0.5) for precision (fewer, more relevant results).
- Higher tau (~2.0–5.0) for recall (more results, lower precision).
- Use `tune_tau()` from `arrowspace_skills` for grid search if you have labelled queries.

## Interpreting scores

Scores are in [0, 1]. Higher means more spectrally coherent with the query. The score blends:
- Proximity in embedding space (semantic similarity)
- Structural alignment in the graph Laplacian (spectral role)

The per-item $$λτ$$ scores (from `aspace.lambdas()` and `aspace.lambdas_sorted()`) can help you understand *why* certain items rank high or low — items with intrinsically high $$λτ$$ will tend to rank higher across queries.

## References

- JOSS paper §3.2: λτ scoring
- Presentation: https://docs.google.com/presentation/d/1f1Zu3FTXltbsXLonflG-7yra4l383B6sbhIgnbx226g/
