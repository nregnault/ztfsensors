# Deprecated pocket-effect correction (v1)

**⚠️ This code is deprecated. Use `ztfsensors.pocket` instead.**

This directory contains the original (v1) pocket-effect correction code,
which has been superseded by the modern equilibrium-function–based pipeline
in `ztfsensors.pocket`.

## Migration guide

| Old (deprecated) | New (recommended) |
|------------------|-------------------|
| `from ztfsensors.deprecated import pocket` | `import ztfsensors.pocket as pocket` |
| `pocket.PocketModel(alpha, cmax, beta, nmax)` | Use `pocket.load_db()` + equilibrium functions |
| `correct.correct_pixels(model, pixels)` | Use `pocket.invert(...)` from the new pipeline |

## Contents

- **`pocket.py`** — Original `PocketModel` class with α, β, cmax, nmax parameters
- **`correct.py`** — Pixel-by-pixel iterative correction routines
- **`_pocket.cpp`** — C++ source for the forward model
- **`_pocket.cpython-*.so`** — Compiled C++ extension (Python backend="cpp")

## Why deprecated?

The v1 model used fixed global parameters (α, β, cmax, nmax) that did not
account for:
- Time-varying detector behavior (readout upgrades, aging)
- Temperature dependence of the equilibrium function
- Robustness to outliers during fitting

The new `ztfsensors.pocket` pipeline addresses all these issues with:
- Equilibrium functions fitted per CCD/quadrant/time-block
- Temperature-dependent models (polynomial or spline basis)
- Robust fitting with outlier rejection
- Parquet-based database for fast lookups
- Built-in visualization (galleries, diagnostics)

See the main package documentation for details.
