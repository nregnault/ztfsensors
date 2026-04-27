"""
Common pytest fixtures for pocket equilibrium model tests.

This module provides shared fixtures that can be used across all test modules.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

# ============================================================================
# Grid and data fixtures
# ============================================================================


@pytest.fixture
def sky_levels():
    """Standard sky level grid for testing."""
    return np.geomspace(50.0, 5000.0, 20)


@pytest.fixture
def temperatures():
    """Standard temperature array for testing."""
    return np.linspace(150.0, 170.0, 20)


@pytest.fixture
def simple_x_grid():
    """Simple logarithmic x grid."""
    return np.array([10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0])


@pytest.fixture
def simple_temp_grid():
    """Simple temperature grid."""
    return np.array([150.0, 155.0, 160.0, 165.0, 170.0])


@pytest.fixture
def jax_grid():
    """Simple JAX-compatible grid for JaxEqFunc testing."""
    x_grid = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
    # y_grid has maximum at the end to avoid clipping during interpolation tests
    y_grid = jnp.array([0.0, 1.0, 2.0, 2.5, 3.0, 3.5], dtype=jnp.float32)
    return x_grid, y_grid


# ============================================================================
# Parameter fixtures
# ============================================================================


@pytest.fixture
def dummy_params_2d():
    """2D parameter array (e.g., for spline model)."""
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


@pytest.fixture
def dummy_params_1d():
    """1D parameter array."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


# ============================================================================
# Synthetic data fixtures
# ============================================================================


@pytest.fixture
def synthetic_data():
    """
    Generate synthetic equilibrium data.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'skylev': sky level values
        - 'temp': temperature values
        - 'overscan': overscan signal values
        - 'params': true parameters used to generate data
    """
    np.random.seed(42)
    n = 100

    skylev = np.random.uniform(50.0, 2000.0, n)
    temp = np.random.uniform(155.0, 165.0, n)

    # Simple linear model: y = a*log(1+x) + b*temp + noise
    true_params = np.array([50.0, 5.0])
    overscan = true_params[0] * np.log1p(skylev) + true_params[1] * temp
    overscan += np.random.normal(0, 2.0, n)  # Add noise

    return {
        "skylev": skylev,
        "temp": temp,
        "overscan": overscan,
        "params": true_params,
    }


@pytest.fixture
def noisy_equilibrium_curve():
    """
    Generate a noisy equilibrium curve for testing fits.

    Returns
    -------
    dict
        Dictionary with 'x', 'y', and 'y_true' (without noise).
    """
    np.random.seed(123)
    x = np.geomspace(50.0, 5000.0, 100)

    # Equilibrium-like curve: rises then saturates
    u = np.log1p(x)
    y_true = 100 * u / (1 + 0.01 * u**2)

    # Add noise
    y = y_true + np.random.normal(0, 2.0, len(x))

    return {"x": x, "y": y, "y_true": y_true}


# ============================================================================
# File system fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing file I/O."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file_prefix(temp_dir):
    """Provide a temporary file prefix for saving results."""
    return temp_dir / "test_output"


# ============================================================================
# Model configuration fixtures
# ============================================================================


@pytest.fixture
def spline_config():
    """Configuration for a spline temperature equilibrium model."""
    return {
        "basis_grid": np.geomspace(50.0, 5000.0, 8),
        "temp_deg": 3,
        "temp_ref": 160.0,
        "temp_scale": 5.0,
        "basis_order": 4,
    }


@pytest.fixture
def poly_config():
    """Configuration for a polynomial temperature equilibrium model."""
    return {
        "p_deg": [2, 2, 2],
        "q_deg": [3, 3],
        "x_knot": 200.0,
        "temp_ref": 160.0,
        "temp_scale": 5.0,
    }


# ============================================================================
# Database fixtures
# ============================================================================


@pytest.fixture
def sample_fit_records():
    """
    Create sample fit records for database testing.

    Returns
    -------
    list[dict]
        List of dictionaries representing fit records.
    """
    return [
        {
            "ccdid": 1,
            "qid": 1,
            "mjd_start": 58000.0,
            "mjd_end": 58100.0,
            "temp_min": 155.0,
            "temp_max": 165.0,
            "params": [[1.0, 2.0], [3.0, 4.0]],
        },
        {
            "ccdid": 1,
            "qid": 1,
            "mjd_start": 58100.0,
            "mjd_end": 58200.0,
            "temp_min": 156.0,
            "temp_max": 166.0,
            "params": [[1.1, 2.1], [3.1, 4.1]],
        },
        {
            "ccdid": 1,
            "qid": 2,
            "mjd_start": 58000.0,
            "mjd_end": 58100.0,
            "temp_min": 155.5,
            "temp_max": 165.5,
            "params": [[5.0, 6.0], [7.0, 8.0]],
        },
    ]


@pytest.fixture
def mjd_intervals():
    """Common MJD intervals for testing."""
    return {
        "early": (58000.0, 58100.0),
        "middle": (58100.0, 58200.0),
        "late": (58200.0, 58300.0),
        "overlapping": (58050.0, 58150.0),  # overlaps with early and middle
    }


# ============================================================================
# Comparison and validation fixtures
# ============================================================================


@pytest.fixture
def atol():
    """Default absolute tolerance for floating point comparisons."""
    return 1e-6


@pytest.fixture
def rtol():
    """Default relative tolerance for floating point comparisons."""
    return 1e-5


@pytest.fixture
def header_with_arrays():
    """Sample header containing numpy arrays (for testing serialization)."""
    return {
        "model_name": "test_model",
        "model_version": "v1",
        "basis_grid": [50.0, 100.0, 500.0, 1000.0],
        "coefficients": [[1.0, 2.0], [3.0, 4.0]],
        "scalar_param": 42.0,
        "string_param": "test",
    }
