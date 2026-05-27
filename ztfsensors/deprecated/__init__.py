"""
Deprecated pocket-effect correction modules.

.. deprecated:: 2024
   These modules are superseded by :mod:`ztfsensors.pocket`, which provides
   a modern equilibrium-function–based correction pipeline.

This package contains the original (v1) pocket-effect correction code:

- :mod:`ztfsensors.deprecated.pocket` — original ``PocketModel`` class
- :mod:`ztfsensors.deprecated.correct` — pixel-by-pixel correction routines
- ``_pocket.cpp`` — C++ backend for the forward model

New code should use::

    from ztfsensors import pocket

instead of::

    from ztfsensors.deprecated import pocket  # old, do not use
"""

import warnings

# Emit a runtime warning when this module is imported
warnings.warn(
    "ztfsensors.deprecated is deprecated. Use ztfsensors.pocket instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the old modules for backward compatibility
from . import correct, pocket

__all__ = ["pocket", "correct"]
