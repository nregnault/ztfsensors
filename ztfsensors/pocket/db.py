from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from .models.base import BaseEquilibriumModel, JaxEqFunc


def _model_from_header(header: dict) -> BaseEquilibriumModel:
    from .models.factory import EquilibriumModelFactory

    return EquilibriumModelFactory.from_header(header)


def _validate_model_header(model: BaseEquilibriumModel, header: dict) -> None:
    model.validate_header(header)


@dataclass
class EqFuncDb:
    """
    Database for equilibrium function parameters.

    Stores fitted equilibrium model parameters organized by CCD ID, quadrant ID,
    and MJD intervals. Provides methods to retrieve parameters and instantiate
    equilibrium functions for specific observations.

    Attributes
    ----------
    df : pl.DataFrame
        DataFrame containing fit records with columns:
        - ccdid, qid: CCD and quadrant identifiers
        - mjd_start, mjd_end: MJD interval [start, end) for which params are valid
        - temp_min, temp_max: temperature range covered by the fit
        - params: flat list of model parameters
    model : BaseEquilibriumModel
        The equilibrium model instance defining the functional form.
    """

    df: pl.DataFrame
    model: BaseEquilibriumModel

    @classmethod
    def open(
        cls,
        prefix: str | Path,
        model: BaseEquilibriumModel | None = None,
    ) -> "EqFuncDb":
        """
        Ouvre une base de données d'équilibre depuis les fichiers .yaml et .parquet.

        Parameters
        ----------
        prefix :
            Chemin sans extension (ex. ``"prod/eq_db"``).
            Les fichiers ``<prefix>.yaml`` et ``<prefix>.parquet`` sont lus.
        model :
            Si fourni, le modèle est validé contre le header YAML pour détecter
            toute incohérence. Si ``None`` (défaut), le modèle est reconstruit
            directement depuis le header.
        """
        prefix = Path(prefix)

        with prefix.with_suffix(".yaml").open("r", encoding="utf-8") as f:
            header = yaml.safe_load(f)

        if model is None:
            model = _model_from_header(header)
        else:
            _validate_model_header(model, header)

        df = pl.read_parquet(prefix.with_suffix(".parquet"))
        return cls(df=df, model=model)

    def select_row(
        self,
        ccdid: int,
        qid: int,
        mjd: float,
    ) -> dict[str, Any]:
        """
        Select the database row matching the given CCD, quadrant, and MJD.

        Parameters
        ----------
        ccdid : int
            CCD identifier.
        qid : int
            Quadrant identifier.
        mjd : float
            Modified Julian Date for which to retrieve parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all columns for the matching row.

        Raises
        ------
        KeyError
            If no matching row is found.
        ValueError
            If multiple overlapping rows are found (indicates database error).
        """
        rows = self.df.filter(
            (pl.col("ccdid") == int(ccdid))
            & (pl.col("qid") == int(qid))
            & (pl.col("mjd_start") <= float(mjd))
            & (pl.col("mjd_end") > float(mjd))
        )

        if rows.height == 0:
            raise KeyError(
                f"No equilibrium block found for ccdid={ccdid}, qid={qid}, mjd={mjd}"
            )
        if rows.height > 1:
            raise ValueError(
                f"Multiple equilibrium blocks found for ccdid={ccdid}, qid={qid}, mjd={mjd}"
            )

        return rows.to_dicts()[0]

    def get_temperature_range(
        self, ccdid: int, qid: int, mjd: float
    ) -> tuple[float, float]:
        """ """
        r = self.select_row(ccdid, qid, mjd)
        return r["temp_min"], r["temp_max"]

    def get_params(
        self,
        ccdid: int,
        qid: int,
        mjd: float,
    ) -> np.ndarray:
        """
        Get validated model parameters for a specific CCD, quadrant, and MJD.

        Parameters
        ----------
        ccdid : int
            CCD identifier.
        qid : int
            Quadrant identifier.
        mjd : float
            Modified Julian Date for which to retrieve parameters.

        Returns
        -------
        np.ndarray
            Validated parameter array with shape matching model.params_shape.

        Raises
        ------
        KeyError
            If no matching parameters are found.
        ValueError
            If parameters have incorrect shape or type.
        """
        row = self.select_row(ccdid=ccdid, qid=qid, mjd=mjd)
        return self.model.validate_params(np.asarray(row["params"]))

    def get_eq_func(
        self,
        ccdid: int,
        qid: int,
        mjd: float,
        ccd_temp: float,
        tabulation_grid=None,
    ) -> JaxEqFunc:
        """
        Create a JIT-compiled equilibrium function for specific observation conditions.

        Retrieves the appropriate parameters and instantiates a fast equilibrium
        function at the given CCD temperature.

        Parameters
        ----------
        ccdid : int
            CCD identifier.
        qid : int
            Quadrant identifier.
        mjd : float
            Modified Julian Date for which to retrieve parameters.
        ccd_temp : float
            CCD temperature in Kelvin at which to evaluate the function.
        tabulation_grid : array_like, optional
            Grid of x values for tabulating the function. If None, uses the
            model's default grid.

        Returns
        -------
        JaxEqFunc
            JIT-compiled equilibrium function ready for fast evaluation.

        Raises
        ------
        KeyError
            If no matching parameters are found in the database.
        ValueError
            If parameters are invalid.
        """
        params = self.get_params(ccdid=ccdid, qid=qid, mjd=mjd)
        return self.model.make_eq_func(
            params=params,
            ccd_temp=ccd_temp,
            tabulation_grid=tabulation_grid,
        )
