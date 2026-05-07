"""
Fitting utilities for equilibrium models.

This module provides classes and functions for fitting equilibrium models to
ZTF CCD sensor data, managing fit results, and diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from saltworks import linearmodels as lm
from sksparse import cholmod

from .models.base import BaseEquilibriumModel

# ---------------------------------------------------------------------------
# Bad-images schema
# ---------------------------------------------------------------------------

# Columns extracted from the training DataFrame to identify bad images.
# Only columns that actually exist in the input DataFrame are kept.
_BAD_IMAGE_WANT_COLS = (
    "filefracday",
    "fieldid",
    "filterid",
    "mjd",
    "skylev",
    "overscan_sum",
)

_BAD_IMAGES_SCHEMA: dict = {
    "filefracday": pl.Int64,
    "fieldid": pl.Int32,
    "filterid": pl.Int16,
    "mjd": pl.Float64,
    "skylev": pl.Float64,
    "overscan_sum": pl.Float64,
    "ccdid": pl.Int16,
    "qid": pl.Int16,
    "mjd_start": pl.Float64,
    "mjd_end": pl.Float64,
}

# ---------------------------------------------------------------------------
# Diagnostics schema
# ---------------------------------------------------------------------------

_DIAG_SCHEMA: dict = {
    "ccdid": pl.Int16,
    "qid": pl.Int16,
    "mjd_start": pl.Float64,
    "mjd_end": pl.Float64,
    "xx": pl.List(pl.Float64),
    "yy": pl.List(pl.Float64),
    "temp": pl.List(pl.Float64),
    "mjd": pl.List(pl.Float64),
    "yhat": pl.List(pl.Float64),
    "resid": pl.List(pl.Float64),
    "bads": pl.List(pl.Boolean),
    # bads[i] = True  →  observation i is an outlier (rejected by robust fit)
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FitRecord:
    """
    Record of a single equilibrium model fit.

    Stores the fitted parameters along with metadata about the data used
    for fitting (CCD/quadrant ID, time interval, temperature range).

    Attributes
    ----------
    ccdid : int
        CCD identifier.
    qid : int
        Quadrant identifier.
    mjd_start : float
        Start of MJD interval (inclusive).
    mjd_end : float
        End of MJD interval (exclusive).
    temp_min : float
        Minimum CCD temperature in the fit data (Kelvin).
    temp_max : float
        Maximum CCD temperature in the fit data (Kelvin).
    params : np.ndarray
        Fitted model parameters with shape matching the model's params_shape.
    """

    ccdid: int
    qid: int
    mjd_start: float
    mjd_end: float
    temp_min: float
    temp_max: float
    params: np.ndarray
    bad_images: pl.DataFrame | None = field(
        default=None,
        compare=False,
        hash=False,
        repr=False,
    )
    """
    DataFrame of outlier images identified during the robust fit, with
    columns ``(filefracday, fieldid, filterid, ccdid, qid, mjd_start, mjd_end)``.
    ``None`` when no outlier detection was performed or no bad images were found.
    """

    def validate(self, model: BaseEquilibriumModel) -> np.ndarray:
        """
        Validate the fit record against a model.

        Checks that MJD and temperature intervals are valid, and that
        parameters have the correct shape for the model.

        Parameters
        ----------
        model : BaseEquilibriumModel
            The equilibrium model to validate against.

        Returns
        -------
        np.ndarray
            Validated parameter array.

        Raises
        ------
        ValueError
            If MJD or temperature intervals are invalid, or if parameters
            have incorrect shape.
        """
        if self.mjd_end <= self.mjd_start:
            raise ValueError(
                f"Invalid MJD interval: [{self.mjd_start}, {self.mjd_end})"
            )
        if self.temp_min >= self.temp_max:
            raise ValueError(
                f"Invalid temperature interval: [{self.temp_min}, {self.temp_max})"
            )
        return model.validate_params(self.params)


@dataclass
class FitResults:
    """
    Collection of equilibrium model fit records.

    Manages multiple FitRecord instances for different CCDs, quadrants, and
    time intervals. Provides serialization to Parquet/YAML format and
    validation to ensure no overlapping intervals.

    Attributes
    ----------
    model : BaseEquilibriumModel
        The equilibrium model used for all fits.
    records : list[FitRecord]
        List of fit records, each representing parameters for a specific
        CCD/quadrant and time interval.
    """

    model: BaseEquilibriumModel
    records: list[FitRecord] = field(default_factory=list)

    def bad_images_to_dataframe(self) -> pl.DataFrame:
        """
        Collect bad-image records from all fits into a single DataFrame.

        Returns
        -------
        pl.DataFrame
            Concatenation of all ``record.bad_images`` DataFrames, with
            schema ``_BAD_IMAGES_SCHEMA``.  Returns an empty DataFrame with
            that schema when no bad images have been recorded.
        """
        empty = pl.DataFrame(schema=_BAD_IMAGES_SCHEMA)
        parts = [
            rec.bad_images
            for rec in self.records
            if rec.bad_images is not None and rec.bad_images.height > 0
        ]
        if not parts:
            return empty
        return pl.concat([empty] + parts)

    def append(self, record: FitRecord) -> None:
        """
        Add a fit record to the collection.

        The record is validated before being added.

        Parameters
        ----------
        record : FitRecord
            Fit record to append.

        Raises
        ------
        ValueError
            If the record fails validation against the model.
        """
        record.validate(self.model)
        self.records.append(record)

    def to_dataframe(self) -> pl.DataFrame:
        """
        Convert all fit records to a Polars DataFrame.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns: ccdid, qid, mjd_start, mjd_end,
            temp_min, temp_max, and params.

        Notes
        -----
        The ``params`` column stores parameters as a flat ``List(Float64)``
        matching the model's ``params_shape``.
        """
        rows = []
        for rec in self.records:
            params = rec.validate(self.model)
            rows.append(
                {
                    "ccdid": rec.ccdid,
                    "qid": rec.qid,
                    "mjd_start": rec.mjd_start,
                    "mjd_end": rec.mjd_end,
                    "temp_min": rec.temp_min,
                    "temp_max": rec.temp_max,
                    "params": params.tolist(),
                }
            )

        return pl.DataFrame(
            rows,
            schema={
                "ccdid": pl.Int16,
                "qid": pl.Int16,
                "mjd_start": pl.Float64,
                "mjd_end": pl.Float64,
                "temp_min": pl.Float64,
                "temp_max": pl.Float64,
                "params": pl.List(pl.Float64),
            },
        )

    def validate(self) -> None:
        """
        Validate the entire collection of fit records.

        Checks for:
        - Duplicate records (same ccdid, qid, mjd_start, mjd_end)
        - Overlapping MJD intervals for the same CCD/quadrant

        Raises
        ------
        ValueError
            If duplicate records are found or if MJD intervals overlap
            for any CCD/quadrant combination.

        Notes
        -----
        The MJD interval convention is [start, end), so intervals can
        touch at endpoints without being considered overlapping.
        """
        df = self.to_dataframe()

        dup = (
            df.group_by(["ccdid", "qid", "mjd_start", "mjd_end"])
            .len()
            .filter(pl.col("len") > 1)
        )
        if dup.height > 0:
            raise ValueError("Duplicate exact blocks found")

        for (ccdid, qid), subdf in df.group_by(["ccdid", "qid"], maintain_order=True):
            subdf = subdf.sort("mjd_start")
            starts = subdf["mjd_start"].to_list()
            ends = subdf["mjd_end"].to_list()

            for i in range(len(starts) - 1):
                if ends[i] > starts[i + 1]:
                    raise ValueError(
                        f"Overlapping MJD intervals for ccdid={ccdid}, qid={qid}: "
                        f"[{starts[i]}, {ends[i]}) overlaps [{starts[i + 1]}, {ends[i + 1]})"
                    )

    def save(self, prefix: str | Path) -> None:
        """
        Save fit results to Parquet and YAML files.

        The fit records are saved to <prefix>.parquet and the model
        configuration to <prefix>.yaml.

        Parameters
        ----------
        prefix : str or Path
            Path prefix for output files (without extension).
            Parent directories are created if they don't exist.

        Raises
        ------
        ValueError
            If validation fails (duplicate or overlapping records).

        Notes
        -----
        The YAML file contains the model configuration needed to reconstruct
        the model instance. The Parquet file contains all fit records.
        """
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        self.validate()

        self.to_dataframe().write_parquet(prefix.with_suffix(".parquet"))

        bads_path = prefix.parent / (prefix.stem + "_bads.parquet")
        self.bad_images_to_dataframe().write_parquet(bads_path)

        # header = {
        #     "model_name": self.model.MODEL_NAME,
        #     "model_version": self.model.MODEL_VERSION,
        #     "basis_grid": self.model.basis_grid.tolist(),
        #     "temp_deg": self.model.temp_deg,
        #     "temp_ref": self.model.temp_ref,
        #     "temp_scale": self.model.temp_scale,
        #     "basis_size": self.model.basis_size,
        #     "basis_order": self.model.basis.order,
        #     "param_layout": "basis_major_high_to_low_temp_power",
        #     "mjd_interval_convention": "[start, end)",
        # }

        header = self.model.as_dict()

        with prefix.with_suffix(".yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                header,
                f,
                sort_keys=False,
                allow_unicode=True,
            )


@dataclass
class FitDiagnostics:
    """
    Diagnostic information for an equilibrium model fit.

    Stores the input data, model predictions, residuals, and outlier flags
    for evaluating fit quality.

    Attributes
    ----------
    model : BaseEquilibriumModel
        The equilibrium model used for fitting.
    record : FitRecord
        The fit record containing the fitted parameters.
    xx : np.ndarray
        Sky level values (input x data).
    yy : np.ndarray
        Observed overscan signal values (input y data).
    temp : np.ndarray
        CCD temperature values for each observation.
    mjd : np.ndarray
        Modified Julian Date for each observation.
    yhat : np.ndarray
        Model predictions at the input x and temperature values.
    resid : np.ndarray
        Residuals: yy - yhat.
    bads_ : np.ndarray or None
        Boolean array indicating outliers (True = bad point).
        None if outlier detection was not performed.
    """

    model: BaseEquilibriumModel
    record: FitRecord

    xx: np.ndarray
    yy: np.ndarray
    temp: np.ndarray
    mjd: np.ndarray

    yhat: np.ndarray
    resid: np.ndarray
    bads_: np.ndarray | None

    @classmethod
    def from_fit_data(
        cls,
        model: BaseEquilibriumModel,
        record: FitRecord,
        xx,
        yy,
        temp,
        mjd,
        bads=None,
    ) -> "FitDiagnostics":
        """
        Create diagnostics from fit data and a fit record.

        Evaluates the model at the input data points and computes residuals.

        Parameters
        ----------
        model : BaseEquilibriumModel
            The equilibrium model.
        record : FitRecord
            Fit record containing the parameters to evaluate.
        xx : array_like
            Sky level values.
        yy : array_like
            Observed overscan signal values.
        temp : array_like
            CCD temperature values.
        mjd : array_like
            Modified Julian Date values.
        bads : array_like, optional
            Boolean array indicating outliers. If None, all points are
            considered good.

        Returns
        -------
        FitDiagnostics
            Diagnostics object with predictions and residuals.

        Raises
        ------
        ValueError
            If input arrays don't have the same shape.
        """
        xx = np.asarray(xx)
        yy = np.asarray(yy)
        temp = np.asarray(temp)
        mjd = np.asarray(mjd)

        if not (xx.shape == yy.shape == temp.shape == mjd.shape):
            raise ValueError("xx, yy, temp, mjd must have the same shape")

        yhat = model.evaluate(x=xx, ccd_temp=temp, params=record.params)
        resid = yy - yhat

        return cls(
            model=model,
            record=record,
            xx=xx,
            yy=yy,
            temp=temp,
            mjd=mjd,
            yhat=np.asarray(yhat),
            resid=np.asarray(resid),
            bads_=bads,
        )

    @property
    def bads(self):
        """
        Get the bad point (outlier) flags.

        Returns
        -------
        np.ndarray
            Boolean array where True indicates an outlier/bad point.
            If no outlier detection was performed, returns an array of
            all True (all points marked as good).
        """
        if self.bads_ is not None:
            return self.bads_
        return np.ones(len(self.mjd), dtype=bool)

    def plot(self, **kwargs):
        """
        Create diagnostic plots for the fit.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the plotting function.

        Returns
        -------
        Figure and axes objects from the plotting function.

        Notes
        -----
        This method imports the plotting function on demand to avoid
        requiring matplotlib as a hard dependency.
        """
        from .plots.eq_plots import plot_equilibrium_fit

        return plot_equilibrium_fit(self, **kwargs)

    def to_row(self) -> dict:
        """
        Serialise this diagnostics object to a dictionary suitable for
        building a :class:`FitDiagnosticsDb` row.

        ``bads_`` is stored as an explicit boolean array (all-``False`` when
        no outlier detection was performed, meaning every observation is good).

        Returns
        -------
        dict
            Keys matching :data:`_DIAG_SCHEMA`.
        """
        bads = self.bads_.tolist() if self.bads_ is not None else [False] * len(self.xx)
        return {
            "ccdid": self.record.ccdid,
            "qid": self.record.qid,
            "mjd_start": self.record.mjd_start,
            "mjd_end": self.record.mjd_end,
            "xx": self.xx.tolist(),
            "yy": self.yy.tolist(),
            "temp": self.temp.tolist(),
            "mjd": self.mjd.tolist(),
            "yhat": self.yhat.tolist(),
            "resid": self.resid.tolist(),
            "bads": bads,
        }


@dataclass
class FitDiagnosticsDb:
    """
    Persistent store for :class:`FitDiagnostics` data.

    Each row of the underlying DataFrame corresponds to one fit
    ``(ccdid, qid, mjd_start, mjd_end)`` and contains the full arrays of
    sky-level values, overscan signals, temperatures, MJDs, model
    predictions, residuals, and outlier flags.

    Typical usage
    -------------
    Save after a fitting run::

        diag_db = FitDiagnosticsDb.from_diagnostics(all_diags)
        diag_db.save(output_dir / "eq_diag.parquet")

    Reload and replot later::

        from ztfsensors.pocket.db import EqFuncDb
        eq_db   = EqFuncDb.open(output_dir / "eq_db")
        diag_db = FitDiagnosticsDb.open(output_dir / "eq_diag.parquet")
        diag    = diag_db.load_fit(ccdid=6, qid=1, mjd_start=58400., eq_db=eq_db)
        fig, _  = diag.plot()
    """

    df: pl.DataFrame

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_diagnostics(cls, diags: list[FitDiagnostics]) -> "FitDiagnosticsDb":
        """
        Build a :class:`FitDiagnosticsDb` from a list of
        :class:`FitDiagnostics` objects.

        Parameters
        ----------
        diags :
            Diagnostics produced by :func:`fit_eq_model`.

        Returns
        -------
        FitDiagnosticsDb
        """
        empty = pl.DataFrame(schema=_DIAG_SCHEMA)
        if not diags:
            return cls(df=empty)
        return cls(
            df=pl.concat(
                [empty, pl.DataFrame([d.to_row() for d in diags], schema=_DIAG_SCHEMA)]
            )
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Write the diagnostics to a Parquet file.

        Parameters
        ----------
        path :
            Output path (created with parent directories if absent).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.write_parquet(path)

    @classmethod
    def open(cls, path: str | Path) -> "FitDiagnosticsDb":
        """
        Load a previously saved diagnostics store.

        Parameters
        ----------
        path :
            Path to a Parquet file written by :meth:`save`.
        """
        return cls(df=pl.read_parquet(path))

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def load_fit(
        self,
        ccdid: int,
        qid: int,
        mjd_start: float,
        eq_db,
    ) -> FitDiagnostics:
        """
        Reconstruct a :class:`FitDiagnostics` for a specific fit.

        The data arrays come from this store; the model and fit record
        (parameters, temperature range, …) are looked up in *eq_db*.

        Parameters
        ----------
        ccdid :
            CCD identifier.
        qid :
            Quadrant identifier.
        mjd_start :
            Start of the MJD validity interval for the desired fit.
        eq_db :
            :class:`~ztfsensors.pocket.db.EqFuncDb` instance that holds the
            corresponding model and parameters.

        Returns
        -------
        FitDiagnostics

        Raises
        ------
        KeyError
            If no diagnostics row matches ``(ccdid, qid, mjd_start)``.
        """
        rows = self.df.filter(
            (pl.col("ccdid") == int(ccdid))
            & (pl.col("qid") == int(qid))
            & (pl.col("mjd_start") == float(mjd_start))
        )
        if rows.height == 0:
            raise KeyError(
                f"No diagnostics for ccdid={ccdid}, qid={qid}, mjd_start={mjd_start}"
            )
        if rows.height > 1:
            raise ValueError(
                f"Multiple diagnostic rows for ccdid={ccdid}, qid={qid}, "
                f"mjd_start={mjd_start}"
            )
        row = rows.row(0, named=True)

        # Reconstruct the FitRecord from eq_db (avoids duplicating params)
        eq_row = eq_db.select_row(ccdid=ccdid, qid=qid, mjd=mjd_start)
        record = FitRecord(
            ccdid=int(eq_row["ccdid"]),
            qid=int(eq_row["qid"]),
            mjd_start=float(eq_row["mjd_start"]),
            mjd_end=float(eq_row["mjd_end"]),
            temp_min=float(eq_row["temp_min"]),
            temp_max=float(eq_row["temp_max"]),
            params=eq_db.model.validate_params(np.asarray(eq_row["params"])),
        )

        bads = np.asarray(row["bads"])
        return FitDiagnostics(
            model=eq_db.model,
            record=record,
            xx=np.asarray(row["xx"]),
            yy=np.asarray(row["yy"]),
            temp=np.asarray(row["temp"]),
            mjd=np.asarray(row["mjd"]),
            yhat=np.asarray(row["yhat"]),
            resid=np.asarray(row["resid"]),
            bads_=bads if bads.any() else None,
        )


def fit_eq_model(
    df,
    ccdid,
    qid,
    eq_model,
    mjd_start: float | None = None,
    mjd_end: float | None = None,
    sky_min: float = 50.0,
    sky_max: float = 10000.0,
    y_min: float = 0.0,
    y_max: float = 800.0,
    logspace_grid: bool = True,
    beta: float = 1.0e-6,
    robust_fit: bool = True,
):
    """
    Fit an equilibrium model to skylevel vs overscan data.

    Filters data for a specific CCD and quadrant, builds the design matrix,
    and solves for model parameters using either standard or robust linear
    regression.

    Parameters
    ----------
    df : pl.DataFrame
        Input data with columns: ccdid, qid, skylev, overscan_sum,
        cryotemp, mjd.
    ccdid : int
        CCD identifier to fit.
    qid : int
        Quadrant identifier to fit.
    eq_model : BaseEquilibriumModel
        The equilibrium model to fit.
    sky_min : float, optional
        Minimum sky level to include in fit (default: 50.0).
    sky_max : float, optional
        Maximum sky level to include in fit (default: 10000.0).
    y_min : float, optional
        Minimum overscan signal to include in fit (default: 0.0).
    y_max : float, optional
        Maximum overscan signal to include in fit (default: 800.0).
    logspace_grid : bool, optional
        Unused parameter (kept for backward compatibility).
    beta : float, optional
        Regularization parameter for Cholesky solver (default: 1e-6).
        Only used if robust_fit=False.
    robust_fit : bool, optional
        If True, use robust regression with outlier rejection (default: True).
        If False, use standard least squares.

    Returns
    -------
    record : FitRecord
        Fit record containing the fitted parameters and metadata.
    diag : FitDiagnostics
        Diagnostics object with predictions, residuals, and outlier flags.

    Notes
    -----
    Robust fitting uses iterative sigma clipping with nsig=5.0 to identify
    and downweight outliers. The robust solver is from the saltworks package.

    The MJD interval in the returned record is [mjd.min(), mjd.max() + 1.0),
    following the convention that intervals are half-open.
    """
    ddf = df.filter(
        (pl.col("ccdid") == ccdid)
        & (pl.col("qid") == qid)
        & pl.col("skylev").is_between(sky_min, sky_max)
        & pl.col("overscan_sum").is_between(y_min, y_max)
    )

    xx = ddf["skylev"].to_numpy()
    yy = ddf["overscan_sum"].to_numpy()
    temp = ddf["cryotemp"].to_numpy()
    mjd = ddf["mjd"].to_numpy()
    # dt, temp_mean, temp_scale = _rescale_temp(temp)

    J = eq_model.build_design_matrix(xx, temp)

    bads = None
    _chi2, _ndof = None, None
    if not robust_fit:
        # We better perform a robust fit here.
        H = J.T @ J
        rhs = J.T @ yy
        fact = cholmod.cholesky(H, beta=beta)
        params = fact(rhs)
    else:
        J = J.tocoo()
        model = lm.LinearModel(J.row, J.col, J.data)
        solver = lm.RobustLinearSolver(model, yy, weights=None, verbose=1)
        params = solver.robust_solution(nsig=5.0)
        logger.info(f"nbads: {solver.bads.sum()}")
        bads = solver.bads
        # chi2 = solver.chi2
        # ndof = solver.ndof()

    _mjd_start = mjd_start if mjd_start is not None else float(ddf["mjd"].min())
    _mjd_end = mjd_end if mjd_end is not None else float(ddf["mjd"].max()) + 1.0

    # Extract bad-image metadata from the training DataFrame.
    bad_images: pl.DataFrame | None = None
    if bads is not None and bads.any():
        keep_cols = [c for c in _BAD_IMAGE_WANT_COLS if c in ddf.columns]
        if keep_cols:
            bad_rows = ddf.filter(pl.Series("_bads", bads)).select(keep_cols)
            bad_images = bad_rows.with_columns(
                pl.lit(ccdid).cast(pl.Int16).alias("ccdid"),
                pl.lit(qid).cast(pl.Int16).alias("qid"),
                pl.lit(_mjd_start).alias("mjd_start"),
                pl.lit(_mjd_end).alias("mjd_end"),
            ).cast(
                {
                    c: t
                    for c, t in _BAD_IMAGES_SCHEMA.items()
                    if c in bad_rows.columns + ["ccdid", "qid", "mjd_start", "mjd_end"]
                }
            )

    record = FitRecord(
        ccdid=ccdid,
        qid=qid,
        mjd_start=_mjd_start,
        mjd_end=_mjd_end,
        temp_min=ddf["cryotemp"].min(),
        temp_max=ddf["cryotemp"].max(),
        params=np.asarray(params),
        bad_images=bad_images,
    )

    diag = FitDiagnostics.from_fit_data(
        model=eq_model,
        record=record,
        xx=xx,
        yy=yy,
        temp=temp,
        mjd=mjd,
        bads=bads,
    )

    return record, diag
