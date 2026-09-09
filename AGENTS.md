# ArrowSpace Skills — Agent Context

This repository contains portable skill definitions for the ArrowSpace spectral vector search library.

## Key facts

- **ArrowSpace** is a vector database with spectral (graph Laplacian) awareness
- **Target version**: arrowspace-rs 0.28.x / pyarrowspace 0.28.x (tau is an alpha-beta blend weight; builder defaults eps=0.5, k=12, topk=6)
- **Rust core**: `arrowspace-rs` at github.com/Mec-iS/arrowspace-rs
- **Python bindings**: `pyarrowspace` at github.com/tuned-org-uk/pyarrowspace, pip-installable as `arrowspace`
- **JOSS paper**: doi:10.21105/joss.09002
- **Author**: Lorenzo Moriondo (Mec-iS) — ArrowSpace creator, Genefold AI founder

## Repo structure

```
SKILL.md                    # opencode skill registration
arrowspace_skills/          # Python package with helper utilities
skills/                     # Portable skill definitions (any agent)
  arrowspace-core.md        # Building an ArrowSpace index
  arrowspace-search.md      # Querying with λτ scoring
  arrowspace-spectral.md    # Spectral analysis and diagnostics
pyproject.toml              # pip installable as arrowspace-skills
```

## Common operations

1. `pip install "arrowspace>=0.28"` to install the core library
2. `ArrowSpaceBuilder().build(params, items)` to create an index
3. `aspace.search(query, gl, tau)` to query
4. Use `arrowspace_skills` utilities for parameter suggestion and tau tuning
5. For corpus-fitted hyperparameters, use `arrowspace_tuner` (`pip install arrowspace-tuner`, then `arrowspace_tuner.tune(items)`)

## Links

- Paper: https://doi.org/10.21105/joss.09002
- Presentation 1: https://docs.google.com/presentation/d/1f1Zu3FTXltbsXLonflG-7yra4l383B6sbhIgnbx226g/
- Presentation 2: https://docs.google.com/presentation/d/1Mtz-_85qpVROnp4U2VrnlSHn0266Z1yc_HfjUtfxYLs
