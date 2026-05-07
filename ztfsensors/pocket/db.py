from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Directory that ships with the package and holds the bundled eq_db files.
_BUNDLED_DATA_DIR = Path(__file__).parent / "data"
_BUNDLED_DB_PREFIX = _BUNDLED_DATA_DIR / "eq_db"

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
    bad_images : pl.DataFrame or None
        Optional DataFrame of outlier images identified during fitting, with
        columns ``(filefracday, fieldid, filterid, ccdid, qid, mjd_start, mjd_end)``.
        Loaded automatically from ``<prefix>_bads.parquet`` by :meth:`open` when
        the file exists.  ``None`` when not available.
    """

    df: pl.DataFrame
    model: BaseEquilibriumModel
    bad_images: pl.DataFrame | None = None
    # Frozen set of (filefracday, ccdid, qid) tuples for O(1) is_bad() lookups.
    # Built lazily in __post_init__ from bad_images.
    _bad_set: frozenset = field(
        default_factory=frozenset, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.bad_images is not None and self.bad_images.height > 0:
            self._bad_set = frozenset(
                self.bad_images.select(["filefracday", "ccdid", "qid"]).iter_rows()
            )
        else:
            self._bad_set = frozenset()

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

        bads_path = prefix.parent / (prefix.stem + "_bads.parquet")
        bad_images = pl.read_parquet(bads_path) if bads_path.exists() else None

        return cls(df=df, model=model, bad_images=bad_images)

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

    # ------------------------------------------------------------------
    # Convenience class-level factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, prefix: str | Path | None = None) -> "EqFuncDb":
        """Load an equilibrium-function database.

        Parameters
        ----------
        prefix :
            Path *without extension* to the database files
            (``<prefix>.yaml`` and ``<prefix>.parquet`` must exist, and
            optionally ``<prefix>_bads.parquet``).
            When *not* given (or ``None``), the database bundled with the
            ``ztfsensors`` package is loaded from
            ``ztfsensors/pocket/data/eq_db.*``.

        Returns
        -------
        EqFuncDb

        Raises
        ------
        FileNotFoundError
            If the requested database files cannot be found.

        Examples
        --------
        Load the bundled (default) database::

            import ztfsensors.pocket as pocket
            db = pocket.load_db()

        Load a custom database::

            db = pocket.load_db("path/to/my_eq_db")
        """
        if prefix is None:
            prefix = _BUNDLED_DB_PREFIX
            yaml_path = prefix.with_suffix(".yaml")
            parquet_path = prefix.with_suffix(".parquet")
            if not yaml_path.exists() or not parquet_path.exists():
                raise FileNotFoundError(
                    "No bundled equilibrium-function database was found at\n"
                    f"  {yaml_path}\n"
                    f"  {parquet_path}\n"
                    "Please generate the database with `fit_equilibrium_function.py` "
                    "and copy the output files to that location, or pass an explicit "
                    "`prefix` to `load_db()`.\n"
                    "See ztfsensors/pocket/data/README.md for details."
                )
        return cls.open(prefix)

    def is_bad(
        self,
        filefracday: int,
        ccdid: int,
        qid: int,
    ) -> bool:
        """
        Check whether an image is flagged as an outlier for a given CCD/quadrant.

        Parameters
        ----------
        filefracday :
            Unique image identifier.
        ccdid :
            CCD identifier.
        qid :
            Quadrant identifier.

        Returns
        -------
        bool
            ``True`` if the image was rejected as an outlier during the
            equilibrium fit for ``(ccdid, qid)``; ``False`` otherwise
            (including when no bad-images table is available).
        """
        return (int(filefracday), int(ccdid), int(qid)) in self._bad_set

    def get_bad_images(
        self,
        ccdid: int | None = None,
        qid: int | None = None,
    ) -> pl.DataFrame:
        """
        Return the bad-images table, optionally filtered by CCD and quadrant.

        Parameters
        ----------
        ccdid :
            If given, keep only rows for this CCD.
        qid :
            If given, keep only rows for this quadrant.

        Returns
        -------
        pl.DataFrame
            Filtered (or full) bad-images DataFrame.  Empty when
            :attr:`bad_images` is ``None``.
        """
        if self.bad_images is None:
            return pl.DataFrame(
                schema={
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
            )
        df = self.bad_images
        if ccdid is not None:
            df = df.filter(pl.col("ccdid") == int(ccdid))
        if qid is not None:
            df = df.filter(pl.col("qid") == int(qid))
        return df

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
