from __future__ import annotations

import numpy as np
from bbf.bspline import BSpline
from scipy import sparse

from .base import BaseEquilibriumModel
from .factory import EquilibriumModelFactory


@EquilibriumModelFactory.register
class SplineTempEqModel(BaseEquilibriumModel):
    """
    Equilibrium model

        f_eq(x, dT) = sum_k P_k(dT) B_k(x)

    where:
      - B_k are spline basis functions
      - P_k are polynomials in dT = T - temp_ref

    Parameter matrix shape:
        (basis_size, temp_deg + 1)

    Column ordering:
        [coeff for dT^deg, ..., coeff for dT, coeff for dT^0]
    """

    MODEL_NAME = "spline_temp_eq"
    MODEL_VERSION = "v1"
    HEADER_FIELDS = (
        "basis_grid",
        "temp_deg",
        "temp_ref",
        "temp_scale",
        "basis_size",
        "basis_order",
        "param_layout",
    )

    def __init__(
        self,
        basis_grid: np.ndarray | None = None,
        temp_deg: int = 5,
        temp_ref: float = 160.0,
        temp_scale: float = 1.0,
        basis_order: int = 4,
        basis_size: int | None = None,
        param_layout: str | None = None,
    ) -> None:
        """
        Initialize a spline-based temperature equilibrium model.

        Parameters
        ----------
        basis_grid : np.ndarray, optional
            Grid of x values defining the spline knots. If None, uses default_basis_grid().
        temp_deg : int, optional
            Degree of the temperature polynomial (default: 5).
        temp_ref : float, optional
            Reference temperature for normalization (default: 160.0 K).
        temp_scale : float, optional
            Temperature scale for normalization (default: 1.0 K).
        basis_order : int, optional
            Order of the B-spline basis (default: 4, cubic splines).
        basis_size : int, optional
            Expected number of basis functions. If provided, validated against computed size.
        param_layout : str, optional
            Parameter layout convention. If None, uses default 'basis_major_high_to_low_temp_power'.

        Raises
        ------
        ValueError
            If temp_scale is zero or if provided basis_size doesn't match computed size.
        """
        if basis_grid is not None:
            self.basis_grid = np.asarray(basis_grid, dtype=float)
        else:
            self.basis_grid = self.default_basis_grid()

        self.basis_order = int(basis_order)
        self.basis = BSpline(self.basis_grid, self.basis_order)

        # Compute actual basis size
        computed_basis_size = int(len(self.basis))

        # Validate basis_size if provided
        if basis_size is not None:
            if int(basis_size) != computed_basis_size:
                raise ValueError(
                    f"Provided basis_size={basis_size} does not match "
                    f"computed basis_size={computed_basis_size}"
                )
        self.basis_size = computed_basis_size

        self.temp_deg = int(temp_deg)
        self.temp_ref = float(temp_ref)
        self.temp_scale = float(temp_scale)

        # Set param_layout with default if not provided
        if param_layout is None:
            self.param_layout = "basis_major_high_to_low_temp_power"
        else:
            self.param_layout = param_layout

        if self.temp_scale == 0:
            raise ValueError("temp_scale must be non-zero")

    @property
    def params_shape(self) -> tuple[int, int]:
        return (self.basis_size, self.temp_deg + 1)

    @property
    def n_model_coeffs(self) -> int:
        """
        Total number of model coefficients.

        Returns
        -------
        int
            Product of basis_size and (temp_deg + 1).
        """
        return self.basis_size * (self.temp_deg + 1)

    def default_basis_grid(self) -> np.ndarray:
        """
        Generate a default logarithmic grid for spline basis knots.

        Returns
        -------
        np.ndarray
            Geometric grid from 50 to 10000 with 10 points.
        """
        return np.geomspace(50.0, 10000.0, 10)

    def params_to_flat(self, params) -> np.ndarray:
        """
        Flatten 2D parameter matrix to 1D array.

        The flattening is consistent with build_design_matrix() column ordering.

        Parameters
        ----------
        params : array_like
            Parameter matrix with shape (basis_size, temp_deg + 1).

        Returns
        -------
        np.ndarray
            Flat array with shape (basis_size * (temp_deg + 1),).
            The ordering is: all basis coeffs for dT^deg, then dT^(deg-1), ..., then dT^0.

        Raises
        ------
        ValueError
            If params shape doesn't match (basis_size, temp_deg + 1).
        """
        params = self.validate_params(params)
        return params.T.reshape(-1)

    def flat_to_params(self, params) -> np.ndarray:
        """
        Reshape flat parameter array to 2D parameter matrix.

        Inverse of params_to_flat().

        Parameters
        ----------
        params : array_like
            Flat parameter array with shape (n_model_coeffs,).

        Returns
        -------
        np.ndarray
            Parameter matrix with shape (basis_size, temp_deg + 1).

        Raises
        ------
        ValueError
            If flat array doesn't have correct length.
        TypeError
            If params are not numeric.
        """
        beta = np.asarray(params)
        if beta.shape != (self.n_model_coeffs,):
            raise ValueError(
                f"Expected flat beta shape {(self.n_model_coeffs,)}, got {beta.shape}"
            )
        if not np.issubdtype(beta.dtype, np.number):
            raise TypeError(f"beta must be numeric, got dtype={beta.dtype}")
        return beta.reshape(self.temp_deg + 1, self.basis_size).T

    def build_design_matrix(self, x, ccd_temp) -> sparse.csr_matrix:
        """
        Build the design matrix for the spline-temperature model.

        The model is: f_eq(x, dT) = sum_k P_k(dT) B_k(x)
        where dT = (ccd_temp - temp_ref) / temp_scale.

        Parameters
        ----------
        x : array_like
            Sky level values.
        ccd_temp : array_like
            CCD temperature values (must have same shape as x).

        Returns
        -------
        sparse.csr_matrix
            Design matrix with shape (len(x), n_model_coeffs).
            Column ordering: [dT^deg * B_k, ..., dT * B_k, B_k] for each basis function.

        Raises
        ------
        ValueError
            If x and ccd_temp have different shapes.
        """
        x = np.asarray(x)
        ccd_temp = np.asarray(ccd_temp)

        if x.shape != ccd_temp.shape:
            raise ValueError("x and ccd_temp must have same shape")

        dt = self.rescale_temp(ccd_temp)

        J0 = self.basis.eval(x).tocoo()
        blocks = []

        for p in range(self.temp_deg, -1, -1):
            Jp = J0.copy()
            if p > 0:
                Jp.data *= dt[Jp.row] ** p
            blocks.append(Jp)

        return sparse.hstack(blocks, format="csr")  # type: ignore
