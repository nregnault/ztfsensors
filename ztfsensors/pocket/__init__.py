from pathlib import Path

from .db import EqFuncDb
from .fit import FitDiagnostics, FitRecord, FitResults, fit_eq_model
from .models.base import JaxEqFunc, NumpyEqFunc
from .pix_distortion import invert, invert_numpy, plot_1d, plot_2d, predict


def correct_pixels(distorted_pixvals, f_eq):
    """Invert the pocket effect on a full 2-D image.

    Dispatches to the numpy or JAX backend based on the type of *f_eq*:

    * :class:`~ztfsensors.pocket.models.base.NumpyEqFunc` → :func:`~ztfsensors.pocket.pix_distortion.invert_numpy` (no JAX at call time)
    * :class:`~ztfsensors.pocket.models.base.JaxEqFunc`   → :func:`~ztfsensors.pocket.pix_distortion.invert`       (JAX vmap/scan)

    Parameters
    ----------
    distorted_pixvals : array_like, shape (nrows, ncols)
        2-D image with pocket-effect distortion.
    f_eq : NumpyEqFunc or JaxEqFunc
        Equilibrium function, as returned by
        :meth:`~ztfsensors.pocket.db.EqFuncDb.get_eq_func`.

    Returns
    -------
    np.ndarray or jax.Array, shape (nrows, ncols)
        Corrected pixel values.
    """
    from .models.base import NumpyEqFunc

    if isinstance(f_eq, NumpyEqFunc):
        return invert_numpy(distorted_pixvals, f_eq)
    return invert(distorted_pixvals, f_eq)


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
    "correct_pixels",
    "invert",
    "invert_numpy",
    "plot_1d",
    "plot_2d",
    "predict",
    # Equilibrium function types
    "NumpyEqFunc",
]
