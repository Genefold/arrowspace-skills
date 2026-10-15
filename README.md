# ArrowSpace Skills

Portable skill definitions for working with **ArrowSpace** — a spectral vector search library that goes beyond cosine similarity by incorporating graph Laplacian eigenstructure.

> ArrowSpace adds a spectral dimension to vector similarity, enabling search that considers both semantic content and structural role in the dataset.

## Quick Start

```bash
uv pip install "arrowspace>=0.30.0"
```

Then:

```python
from arrowspace import ArrowSpaceBuilder
import numpy as np

items = np.array([[0.1, 0.2, 0.3], [0.0, 0.5, 0.1], [0.9, 0.1, 0.0]], dtype=np.float64)
graph_params = {"eps": 0.5, "k": 12, "topk": 6, "p": 2.0, "sigma": None}
aspace, gl = ArrowSpaceBuilder().build(graph_params, items)

query = np.array([0.05, 0.2, 0.25], dtype=np.float64)
hits = aspace.search(query, gl, tau=1.0)
print(hits)  # [(0, 0.99), (1, 0.76), (2, 0.22)]
```

The `graph_params` above are fallback defaults. Use [`arrowspace_tuner`](https://pypi.org/project/arrowspace-tuner/) to derive the base graph parameters from a representative sample of up to 10,000 items, then build the index over the full corpus:

```python
from arrowspace_tuner import EpsTuner

# The tuner samples internally; fitting always returns graph parameters for
# the final full-corpus build. `best_tau` remains a query-time setting.
tuner = EpsTuner(n_trials=15, seed=3407, sample_n=min(10_000, len(items)))
graph_params = tuner.fit(items)  # eps, k, topk, p, sigma
aspace, gl = ArrowSpaceBuilder().build(graph_params, items)
```

## Resources

| Resource | Link |
|---|---|
| **JOSS paper** | [doi:10.21105/joss.09002](https://doi.org/10.21105/joss.09002) |
| **Rust library** | [arrowspace-rs v0.30.0](https://github.com/tuned-org-uk/arrowspace-rs/releases/tag/v0.30.0) |
| **Python bindings** | [github.com/tuned-org-uk/pyarrowspace](https://github.com/tuned-org-uk/pyarrowspace) |
| **Python bindings** | [pyarrowspace v0.30.0](https://github.com/tuned-org-uk/pyarrowspace/releases/tag/v0.30.0); `uv pip install "arrowspace>=0.30.0"` |
| **crates.io** | `cargo add arrowspace` |
| **Presentation 1** | [Spectral Indexing overview](https://docs.google.com/presentation/d/1f1Zu3FTXltbsXLonflG-7yra4l383B6sbhIgnbx226g/) |
| **Presentation 2** | [ArrowSpace deep-dive](https://docs.google.com/presentation/d/1Mtz-_85qpVROnp4U2VrnlSHn0266Z1yc_HfjUtfxYLs) |
| **Design article** | [tuned.org.uk — Part 2: Semantic Basins](https://www.tuned.org.uk/posts/020_arrowspace_semantic_basins_part2) |
| **Complex examples** | [pyarrowspace test suite](https://github.com/tuned-org-uk/pyarrowspace/tree/main/tests) |
| **Hyperparameter tuner** | [arrowspace_tuner on PyPI](https://pypi.org/project/arrowspace-tuner/) |

## Skills

This repo contains skill definitions that teach AI agents how to use ArrowSpace:

| Skill | File | What it covers |
|---|---|---|
| **ArrowSpace Core** | `skills/arrowspace-core.md` | Builder, configuration, building signal graphs |
| **Vector Search** | `skills/arrowspace-search.md` | Querying, λτ scoring, search parameters |
| **Spectral Analysis** | `skills/arrowspace-spectral.md` | Spectral methods, graph characterisation, diffusion |

### Adding to an AI Agent

**OpenCode** — symlink or copy `SKILL.md` into `.opencode/skills/`:

```bash
cp SKILL.md .opencode/skills/arrowspace/SKILL.md
```

**Claude Code** — reference in `CLAUDE.md` or add to project instructions.

**Copilot / Cursor** — include the relevant `skills/*.md` content in your custom instructions.

**Any agent** — point it at this repo: "You have access to the ArrowSpace skills at `skills/`."

## Helper Scripts

Supporting Python scripts live in `skills/scripts/` — no install needed. Ensure the core library is available, then import them directly:

```bash
uv pip install "arrowspace>=0.30.0"
```

```python
import sys
sys.path.insert(0, "skills/scripts")

from builder import (
    suggest_params,              # fallback heuristic; prefer arrowspace_tuner
    build_index,                 # one-shot index builder
)
from search import (
    tune_tau,                    # grid search over alpha-blend weight
    search_with_recall,          # query with result cap
)
from spectral import (
    item_lambdas,                # per-item lambda-tau scores
    sorted_lambdas,              # lambda-tau-ranked item list
    explain_spectral_properties, # eigendecomposition diagnostics
    spectral_summary,            # human-readable spectral report
)
```

See `skills/scripts/` for the helper functions.

## License

MIT OR Apache-2.0 — same as `arrowspace-rs`.
