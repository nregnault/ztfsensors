from pathlib import Path

from .db import EqFuncDb
from .fit import FitDiagnostics, FitRecord, FitResults, fit_eq_model
from .pix_distortion import invert, plot_1d, plot_2d, predict


def load_db(prefix: "str | Path | None" = None) -> EqFuncDb:
    """Load an equilibrium-function database.

    Parameters
    ----------
    prefix :
        Path *without extension* to the database files
        (``<prefix>.yaml`` and ``<prefix>.parquet`` must exist, and
        optionally ``<prefix>_bads.parquet``).
        When ``None`` (default), the database bundled with the
        ``ztfsensors`` package is used.

    Returns
    -------
    EqFuncDb

    Examples
    --------
    ::

        import ztfsensors.pocket as pocket

        # load the bundled database
        db = pocket.load_db()

        # load a custom database
        db = pocket.load_db("path/to/my_eq_db")
    """
    return EqFuncDb.load(prefix)


__all__ = [
    # DB
    "EqFuncDb",
    "load_db",
    # Fitting pipeline
    "fit_eq_model",
    "FitRecord",
    "FitResults",
    "FitDiagnostics",
    # Pixel distortion utilities
    "invert",
    "plot_1d",
    "plot_2d",
    "predict",
]
