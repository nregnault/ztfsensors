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

    record = FitRecord(
        ccdid=ccdid,
        qid=qid,
        mjd_start=mjd_start if mjd_start is not None else float(ddf["mjd"].min()),
        mjd_end=mjd_end if mjd_end is not None else float(ddf["mjd"].max()) + 1.0,
        temp_min=ddf["cryotemp"].min(),
        temp_max=ddf["cryotemp"].max(),
        params=np.asarray(params),
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
