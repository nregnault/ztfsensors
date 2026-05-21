"""
Base equilibrium model: contract and common methods.

This module defines the abstract base class for equilibrium models used to
characterize the pocket effect in ZTF CCD sensors. The equilibrium function
relates the observed overscan signal to the sky level and CCD temperature.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse


class JaxEqFunc(eqx.Module):
    """
    JAX-compatible equilibrium function with JIT compilation.

    This class provides a fast, JIT-compiled equilibrium function by tabulating
    the model on a grid and using linear interpolation. Input values are clipped
    to avoid extrapolation beyond the peak of the equilibrium curve.

    Attributes
    ----------
    x_grid : jax.Array
        Grid of x values (sky levels) for tabulation.
    y_grid : jax.Array
        Corresponding y values (overscan signal) at each grid point.
    x_max : jax.Array
        x value at the maximum of y_grid (used for clipping).
    """

    x_grid: jax.Array
    y_grid: jax.Array
    x_max: jax.Array

    def __init__(self, x_grid, y_grid):
        """
        Initialize the equilibrium function from tabulated values.

        Parameters
        ----------
        x_grid : array_like
            Grid of x values (sky levels).
        y_grid : array_like
            Corresponding y values (overscan signal).
        """
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.x_max = self.x_grid[jnp.argmax(self.y_grid)]

    @eqx.filter_jit
    def __call__(self, x):
        """
        Evaluate the equilibrium function at given x value(s).

        Values beyond x_max are clipped to avoid extrapolation past the
        equilibrium curve's peak.

        Parameters
        ----------
        x : array_like
            Sky level value(s) at which to evaluate the function.

        Returns
        -------
        jax.Array
            Interpolated overscan signal value(s).
        """
        x_arr = jnp.asarray(x, jnp.float32)
        x_clipped = jnp.clip(x_arr, self.x_grid[0], self.x_max)
        return jnp.interp(x_clipped, self.x_grid, self.y_grid)


class NumpyEqFunc:
    """Pure-numpy equilibrium function via linear interpolation on a tabulated grid.

    Drop-in replacement for :class:`JaxEqFunc` for the inference path.
    No JAX dependency at call time.

    Attributes
    ----------
    x_grid : np.ndarray
        Grid of x values (sky levels) for tabulation.
    y_grid : np.ndarray
        Corresponding y values (overscan signal) at each grid point.
    x_max : float
        x value at the maximum of y_grid (used for clipping, same as JaxEqFunc).
    """

    def __init__(self, x_grid, y_grid):
        self.x_grid = np.asarray(x_grid, dtype=np.float32)
        self.y_grid = np.asarray(y_grid, dtype=np.float32)
        self.x_max = self.x_grid[np.argmax(self.y_grid)]

    def __call__(self, x):
        """Evaluate the equilibrium function at x (any shape).

        Values beyond x_max are clipped identically to JaxEqFunc.

        Parameters
        ----------
        x : array_like
            Sky level value(s).  np.interp handles any shape natively.

        Returns
        -------
        np.ndarray
            Interpolated overscan signal value(s), same shape as x.
        """
        x_arr = np.asarray(x, dtype=np.float32)
        x_clipped = np.clip(x_arr, self.x_grid[0], self.x_max)
        # np.interp always returns float64 regardless of input dtypes; cast back
        # to float32 to match JaxEqFunc's behaviour and keep invert_numpy in f32.
        return np.interp(x_clipped, self.x_grid, self.y_grid).astype(np.float32)


class BaseEquilibriumModel(ABC):
    """
    Abstract base class for equilibrium models.

    Defines the interface for equilibrium models that characterize the pocket
    effect in ZTF CCD sensors. Subclasses must implement params_shape and
    build_design_matrix.

    Class Attributes
    ----------------
    MODEL_NAME : str
        Unique identifier for the model type.
    MODEL_VERSION : str
        Version string for the model.
    HEADER_FIELDS : tuple[str, ...]
        Tuple of attribute names to serialize in the model header.
    """

    MODEL_NAME: ClassVar[str]
    MODEL_VERSION: ClassVar[str]
    HEADER_FIELDS: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def _validate_identity(
        cls,
        header: dict[str, Any],
    ) -> None:
        """
        Validate that header model name and version match this class.

        Parameters
        ----------
        header : dict[str, Any]
            Header dictionary to validate.

        Raises
        ------
        ValueError
            If model_name or model_version doesn't match.
        """
        if header.get("model_name") != cls.MODEL_NAME:
            raise ValueError(
                f"Wrong model name: {header.get('model_name')} != {cls.MODEL_NAME}"
            )
        if header.get("model_version") != cls.MODEL_VERSION:
            raise ValueError(
                f"Wrong model version: {header.get('model_version')} != {cls.MODEL_VERSION}"
            )

    @classmethod
    def from_header(
        cls,
        header: dict[str, Any],
    ) -> BaseEquilibriumModel:
        """
        Instantiate a model instance from a serialized header.

        Parameters
        ----------
        header : dict[str, Any]
            Dictionary containing model configuration. Must include
            'model_name', 'model_version', and all fields in HEADER_FIELDS.

        Returns
        -------
        BaseEquilibriumModel
            New instance of the model with parameters from the header.

        Raises
        ------
        ValueError
            If header model name/version doesn't match this class.
        KeyError
            If required fields are missing from the header.
        """
        cls._validate_identity(header)
        kwargs = {name: header[name] for name in cls.HEADER_FIELDS}
        return cls(**kwargs)

    def as_dict(self) -> dict[str, Any]:
        """
        Serialize the model configuration to a dictionary.

        All numpy array values are converted to plain Python lists so the
        result is safe to pass to ``yaml.safe_dump`` without a custom
        representer.

        Returns
        -------
        dict[str, Any]
            Dictionary containing model name, version, MJD convention,
            and all attributes specified in HEADER_FIELDS.
        """
        out = {
            "model_name": self.MODEL_NAME,
            "model_version": self.MODEL_VERSION,
            "mjd_interval_convention": "[start, end)",
        }
        for name in self.HEADER_FIELDS:
            value = getattr(self, name)
            # yaml.safe_dump cannot serialize numpy arrays; convert to list.
            if isinstance(value, np.ndarray):
                value = value.tolist()
            out[name] = value
        return out

    def validate_header(self, header: dict[str, Any]) -> None:
        """
        Validate that a header can reconstruct this model instance.

        Compares the header against the current instance to ensure all
        attributes match.

        Parameters
        ----------
        header : dict[str, Any]
            Header dictionary to validate.

        Raises
        ------
        ValueError
            If any header values don't match the current instance.
        """
        self._validate_identity(header)
        current = self.as_dict()
        for key in self.HEADER_FIELDS:
            expected = current[key]
            actual = header[key]
            self._compare_header_value(key, actual, expected)
        return

    def _compare_header_value(self, key: str, actual: Any, expected: Any) -> None:
        """
        Compare a single header value against the expected value.

        Handles special cases for numpy arrays and floats.

        Parameters
        ----------
        key : str
            Name of the attribute being compared.
        actual : Any
            Value from the header.
        expected : Any
            Expected value from the current instance.

        Raises
        ------
        ValueError
            If values don't match within tolerance.
        """
        if isinstance(expected, np.ndarray):
            actual_arr = np.asarray(actual)
            if actual_arr.shape != expected.shape or not np.allclose(
                actual_arr, expected
            ):
                raise ValueError(f"{key} mismatch")
            return

        if isinstance(expected, float):
            if not math.isclose(float(actual), expected):
                raise ValueError(f"{key} mismatch")
            return

        if actual != expected:
            raise ValueError(f"{key} mismatch: {actual} != {expected}")

    @property
    @abstractmethod
    def params_shape(self) -> tuple[int, ...]:
        """
        Shape of the parameter array.

        Returns
        -------
        tuple[int, ...]
            Shape tuple for the parameter array.
        """
        ...

    def validate_params(self, params) -> np.ndarray:
        """
        Validate and convert parameters to the correct shape and type.

        Parameters
        ----------
        params : array_like
            Model parameters to validate.

        Returns
        -------
        np.ndarray
            Validated parameter array.

        Raises
        ------
        ValueError
            If parameter shape doesn't match params_shape.
        TypeError
            If parameters are not numeric.
        """
        arr = np.asarray(params)
        if arr.shape != self.params_shape:
            raise ValueError(f"expected {self.params_shape} got {arr.shape}")
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"params must be numeric, got dtype={arr.dtype}")
        return arr

    def rescale_temp(self, temp):
        """
        Rescale temperature relative to reference value and scale.

        If the model has temp_ref and temp_scale attributes, returns:
            (temp - temp_ref) / temp_scale

        Otherwise returns temp unchanged.

        Parameters
        ----------
        temp : array_like
            CCD temperature value(s) in Kelvin.

        Returns
        -------
        array_like
            Rescaled temperature value(s).
        """
        if not hasattr(self, "temp_ref") or not hasattr(self, "temp_scale"):
            return temp
        return (temp - self.temp_ref) / self.temp_scale  # type: ignore

    @abstractmethod
    def build_design_matrix(self, x, ccd_temp) -> Union[sparse.spmatrix, np.ndarray]:
        """
        Build the design matrix for the model.

        Parameters
        ----------
        x : array_like
            Sky level values.
        ccd_temp : array_like
            CCD temperature values (must have same shape as x).

        Returns
        -------
        sparse matrix or ndarray
            Design matrix with shape (len(x), n_params).
        """
        ...

    def evaluate(self, x, ccd_temp, params) -> np.ndarray:
        """
        Evaluate the model at given x and temperature values.

        Parameters
        ----------
        x : array_like
            Sky level values.
        ccd_temp : array_like or float
            CCD temperature value(s). If scalar, broadcast to all x values.
        params : array_like
            Model parameters with shape matching params_shape.

        Returns
        -------
        np.ndarray
            Model predictions at the given x and temperature values.

        Raises
        ------
        ValueError
            If x and ccd_temp have incompatible shapes or if params are invalid.
        """
        x = np.asarray(x)
        beta = self.validate_params(params)
        if np.ndim(ccd_temp) == 0:
            ccd_temp = np.full(len(x), ccd_temp)
        else:
            ccd_temp = np.asarray(ccd_temp)

        if x.shape != ccd_temp.shape:
            raise ValueError("x and ccd_temp must have same shape")

        J = self.build_design_matrix(x, ccd_temp)
        y = J @ beta

        return np.asarray(y)

    def default_tabulation_grid(
        self,
        n: int = 100,
        xmin: float = 50.0,
        xmax: float = 5000.0,
    ) -> np.ndarray:
        """
        Generate a default grid for tabulating the equilibrium function.

        Returns a logarithmic (geometric) grid if xmin > 0, otherwise linear.

        Parameters
        ----------
        n : int, optional
            Number of grid points (default: 100).
        xmin : float, optional
            Minimum x value (default: 50.0).
        xmax : float, optional
            Maximum x value (default: 5000.0).

        Returns
        -------
        np.ndarray
            Grid of x values for tabulation.

        Raises
        ------
        ValueError
            If xmin >= xmax.
        """
        if xmin >= xmax:
            raise ValueError(f"xmin={xmin} >= xmax={xmax}")
        if xmin <= 0:
            return np.linspace(xmin, xmax, n)

        return np.geomspace(xmin, xmax, n)

    def make_eq_func(
        self, params, ccd_temp: float, tabulation_grid=None, backend: str = "numpy"
    ):
        """
        Create an equilibrium function at a specific temperature.

        Tabulates the model on a grid and returns a fast interpolating function.

        Parameters
        ----------
        params : array_like
            Model parameters with shape matching params_shape.
        ccd_temp : float
            CCD temperature at which to evaluate the function.
        tabulation_grid : array_like, optional
            Grid of x values for tabulation. If None, uses default_tabulation_grid().
        backend : str, optional
            Which backend to use for the returned callable.
            ``'numpy'`` (default) returns a :class:`NumpyEqFunc` (no JAX at call time).
            ``'jax'`` returns a :class:`JaxEqFunc` (JIT-compiled via equinox).

        Returns
        -------
        NumpyEqFunc or JaxEqFunc
            Equilibrium function ready for evaluation.

        Raises
        ------
        ValueError
            If params have incorrect shape.
        """
        params = self.validate_params(params)

        if tabulation_grid is None:
            tabulation_grid = self.default_tabulation_grid(n=100)
        else:
            tabulation_grid = np.asarray(tabulation_grid)

        y_grid = self.evaluate(x=tabulation_grid, ccd_temp=ccd_temp, params=params)

        if backend == "numpy":
            return NumpyEqFunc(x_grid=tabulation_grid, y_grid=y_grid)
        elif backend == "jax":
            return JaxEqFunc(
                x_grid=jnp.asarray(tabulation_grid, dtype=jnp.float32),
                y_grid=jnp.asarray(y_grid, dtype=jnp.float32),
            )
        else:
            raise ValueError(f"Unknown backend {backend!r}: expected 'numpy' or 'jax'.")
