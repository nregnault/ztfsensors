"""
ZTF Sensors package.

Submodules
----------
pocket : Modern equilibrium-function–based pocket-effect correction
deprecated : Original (v1) pocket-effect correction (deprecated)
config : Configuration utilities
test : Test utilities

Usage
-----
::

    import ztfsensors.pocket as pocket
    db = pocket.load_db()
    # ...

See Also
--------
For the deprecated v1 pocket model::

    from ztfsensors.deprecated import pocket  # shows DeprecationWarning
"""

# Version info (if available)
try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = ["pocket", "deprecated", "config", "test"]
