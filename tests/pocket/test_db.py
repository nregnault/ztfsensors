"""
Tests for EqFuncDb.

This module tests the equilibrium function database class, including
loading, querying, and retrieving parameters and equilibrium functions.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from ztfsensors.pocket.db import EqFuncDb, _model_from_header, _validate_model_header
from ztfsensors.pocket.models.base import JaxEqFunc
from ztfsensors.pocket.models.poly_temp_eq_model import PolyTempEqModel
from ztfsensors.pocket.models.spline_temp_eq_model import SplineTempEqModel

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing file I/O."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_spline_model():
    """Create a sample spline model for testing."""
    return SplineTempEqModel(
        basis_grid=np.geomspace(50.0, 5000.0, 8),
        temp_deg=3,
        temp_ref=160.0,
        temp_scale=1.0,
        basis_order=4,
    )


@pytest.fixture
def sample_poly_model():
    """Create a sample poly model for testing."""
    return PolyTempEqModel(
        p_deg=[2, 2, 2],
        q_deg=[3, 3],
        x_knot=200.0,
        temp_ref=160.0,
        temp_scale=1.0,
    )


@pytest.fixture
def sample_db_data(sample_spline_model):
    """Create sample database data."""

    def _flat(model):
        return np.random.randn(*model.params_shape).tolist()

    records = [
        {
            "ccdid": 1,
            "qid": 1,
            "mjd_start": 58000.0,
            "mjd_end": 58100.0,
            "temp_min": 155.0,
            "temp_max": 165.0,
            "params": _flat(sample_spline_model),
        },
        {
            "ccdid": 1,
            "qid": 1,
            "mjd_start": 58100.0,
            "mjd_end": 58200.0,
            "temp_min": 156.0,
            "temp_max": 166.0,
            "params": _flat(sample_spline_model),
        },
        {
            "ccdid": 1,
            "qid": 2,
            "mjd_start": 58000.0,
            "mjd_end": 58150.0,
            "temp_min": 155.5,
            "temp_max": 165.5,
            "params": _flat(sample_spline_model),
        },
        {
            "ccdid": 2,
            "qid": 1,
            "mjd_start": 58000.0,
            "mjd_end": 58100.0,
            "temp_min": 154.0,
            "temp_max": 164.0,
            "params": _flat(sample_spline_model),
        },
    ]

    df = pl.DataFrame(records)
    return df


@pytest.fixture
def sample_poly_db_data(sample_poly_model):
    """Create sample database data with poly model for get_eq_func tests."""

    def _flat(model):
        return np.random.randn(*model.params_shape).tolist()

    records = [
        {
            "ccdid": 1,
            "qid": 1,
            "mjd_start": 58000.0,
            "mjd_end": 58100.0,
            "temp_min": 155.0,
            "temp_max": 165.0,
            "params": _flat(sample_poly_model),
        },
        {
            "ccdid": 1,
            "qid": 1,
            "mjd_start": 58100.0,
            "mjd_end": 58200.0,
            "temp_min": 156.0,
            "temp_max": 166.0,
            "params": _flat(sample_poly_model),
        },
    ]

    df = pl.DataFrame(records)
    return df


@pytest.fixture
def sample_poly_db(sample_poly_model, sample_poly_db_data):
    """Create a sample EqFuncDb instance with poly model."""
    return EqFuncDb(df=sample_poly_db_data, model=sample_poly_model)


@pytest.fixture
def sample_db_files(temp_dir, sample_spline_model, sample_db_data):
    """Create sample database files on disk."""
    prefix = temp_dir / "test_eq_db"

    # Write parquet file
    sample_db_data.write_parquet(prefix.with_suffix(".parquet"))

    # Write YAML header (as_dict() already converts numpy arrays to lists)
    header = sample_spline_model.as_dict()
    with prefix.with_suffix(".yaml").open("w") as f:
        yaml.safe_dump(header, f)

    return prefix


@pytest.fixture
def sample_db(sample_spline_model, sample_db_data):
    """Create a sample EqFuncDb instance."""
    return EqFuncDb(df=sample_db_data, model=sample_spline_model)


# ============================================================================
# Test Helper Functions
# ============================================================================


class TestHelperFunctions:
    """Test helper functions for model creation and validation."""

    def test_model_from_header_spline(self):
        """Test creating a spline model from header."""
        # First create a model to get the correct basis_size
        temp_model = SplineTempEqModel(
            basis_grid=np.array([50.0, 100.0, 500.0, 1000.0, 5000.0]),
            basis_order=4,
        )

        header = {
            "model_name": "spline_temp_eq",
            "model_version": "v1",
            "basis_grid": [50.0, 100.0, 500.0, 1000.0, 5000.0],
            "temp_deg": 3,
            "temp_ref": 160.0,
            "temp_scale": 1.0,
            "basis_order": 4,
            "basis_size": temp_model.basis_size,
            "param_layout": "basis_major_high_to_low_temp_power",
        }

        model = _model_from_header(header)

        assert isinstance(model, SplineTempEqModel)
        assert model.temp_deg == 3
        assert model.temp_ref == 160.0

    def test_model_from_header_poly(self):
        """Test creating a poly model from header."""
        header = {
            "model_name": "poly_temp_eq",
            "model_version": "v1",
            "p_deg": [2, 2, 2],
            "q_deg": [3, 3],
            "x_knot": 200.0,
            "temp_ref": 160.0,
            "temp_scale": 1.0,
        }

        model = _model_from_header(header)

        assert isinstance(model, PolyTempEqModel)
        assert model.p_deg == [2, 2, 2]
        assert model.x_knot == 200.0

    def test_validate_model_header_correct(self, sample_spline_model):
        """Test validating correct header."""
        header = sample_spline_model.as_dict()

        # Should not raise any error
        _validate_model_header(sample_spline_model, header)

    def test_validate_model_header_wrong_value(self, sample_spline_model):
        """Test that wrong header values raise error."""
        header = sample_spline_model.as_dict()
        header["temp_ref"] = 999.0  # Wrong value

        with pytest.raises(ValueError):
            _validate_model_header(sample_spline_model, header)


# ============================================================================
# Test EqFuncDb Initialization
# ============================================================================


class TestEqFuncDbInit:
    """Test EqFuncDb initialization."""

    def test_init_with_dataframe_and_model(self, sample_db_data, sample_spline_model):
        """Test initialization with DataFrame and model."""
        db = EqFuncDb(df=sample_db_data, model=sample_spline_model)

        assert isinstance(db.df, pl.DataFrame)
        assert isinstance(db.model, SplineTempEqModel)
        assert db.df.height == 4  # 4 records

    def test_init_preserves_dataframe(self, sample_db_data, sample_spline_model):
        """Test that initialization preserves the DataFrame."""
        db = EqFuncDb(df=sample_db_data, model=sample_spline_model)

        assert db.df.equals(sample_db_data)


# ============================================================================
# Test EqFuncDb.open
# ============================================================================


class TestEqFuncDbOpen:
    """Test the open class method."""

    def test_open_from_files(self, sample_db_files):
        """Test opening database from files."""
        db = EqFuncDb.open(sample_db_files)

        assert isinstance(db, EqFuncDb)
        assert isinstance(db.df, pl.DataFrame)
        assert db.df.height > 0
        assert isinstance(db.model, SplineTempEqModel)

    def test_open_with_model_validation(self, sample_db_files, sample_spline_model):
        """Test opening with model validation."""
        db = EqFuncDb.open(sample_db_files, model=sample_spline_model)

        assert isinstance(db, EqFuncDb)
        assert db.model is sample_spline_model

    def test_open_with_string_path(self, sample_db_files):
        """Test opening with string path instead of Path."""
        db = EqFuncDb.open(str(sample_db_files))

        assert isinstance(db, EqFuncDb)

    def test_open_nonexistent_file(self, temp_dir):
        """Test that opening nonexistent file raises error."""
        nonexistent = temp_dir / "nonexistent"

        with pytest.raises(FileNotFoundError):
            EqFuncDb.open(nonexistent)

    def test_open_reconstructs_model(self, sample_db_files):
        """Test that open reconstructs model from header when model=None."""
        db = EqFuncDb.open(sample_db_files, model=None)

        # Model should be reconstructed from YAML
        assert db.model is not None
        assert isinstance(db.model, SplineTempEqModel)


# ============================================================================
# Test select_row
# ============================================================================


class TestSelectRow:
    """Test the select_row method."""

    def test_select_row_valid(self, sample_db):
        """Test selecting a valid row."""
        row = sample_db.select_row(ccdid=1, qid=1, mjd=58050.0)

        assert row["ccdid"] == 1
        assert row["qid"] == 1
        assert row["mjd_start"] <= 58050.0 < row["mjd_end"]

    def test_select_row_different_ccd(self, sample_db):
        """Test selecting row for different CCD."""
        row = sample_db.select_row(ccdid=2, qid=1, mjd=58050.0)

        assert row["ccdid"] == 2
        assert row["qid"] == 1

    def test_select_row_different_quadrant(self, sample_db):
        """Test selecting row for different quadrant."""
        row = sample_db.select_row(ccdid=1, qid=2, mjd=58050.0)

        assert row["ccdid"] == 1
        assert row["qid"] == 2

    def test_select_row_different_mjd_interval(self, sample_db):
        """Test selecting row for different MJD interval."""
        # First interval
        row1 = sample_db.select_row(ccdid=1, qid=1, mjd=58050.0)
        assert row1["mjd_start"] == 58000.0
        assert row1["mjd_end"] == 58100.0

        # Second interval
        row2 = sample_db.select_row(ccdid=1, qid=1, mjd=58150.0)
        assert row2["mjd_start"] == 58100.0
        assert row2["mjd_end"] == 58200.0

    def test_select_row_at_start_boundary(self, sample_db):
        """Test selecting row at start boundary (inclusive)."""
        row = sample_db.select_row(ccdid=1, qid=1, mjd=58000.0)

        assert row["mjd_start"] == 58000.0

    def test_select_row_at_end_boundary(self, sample_db):
        """Test selecting row at end boundary (exclusive)."""
        # mjd=58100.0 should NOT match [58000, 58100) but should match [58100, 58200)
        row = sample_db.select_row(ccdid=1, qid=1, mjd=58100.0)

        assert row["mjd_start"] == 58100.0
        assert row["mjd_end"] == 58200.0

    def test_select_row_no_match(self, sample_db):
        """Test that selecting non-existent row raises KeyError."""
        with pytest.raises(KeyError, match="No equilibrium block found"):
            sample_db.select_row(ccdid=999, qid=1, mjd=58050.0)

    def test_select_row_mjd_out_of_range(self, sample_db):
        """Test that MJD outside all intervals raises KeyError."""
        with pytest.raises(KeyError, match="No equilibrium block found"):
            sample_db.select_row(ccdid=1, qid=1, mjd=59000.0)

    def test_select_row_returns_dict(self, sample_db):
        """Test that select_row returns a dictionary."""
        row = sample_db.select_row(ccdid=1, qid=1, mjd=58050.0)

        assert isinstance(row, dict)
        assert "ccdid" in row
        assert "qid" in row
        assert "mjd_start" in row
        assert "mjd_end" in row
        assert "params" in row


# ============================================================================
# Test get_params
# ============================================================================


class TestGetParams:
    """Test the get_params method."""

    def test_get_params_valid(self, sample_db):
        """Test getting valid parameters."""
        params = sample_db.get_params(ccdid=1, qid=1, mjd=58050.0)

        assert isinstance(params, np.ndarray)
        assert params.shape == sample_db.model.params_shape

    def test_get_params_different_ccd(self, sample_db):
        """Test getting parameters for different CCD."""
        params1 = sample_db.get_params(ccdid=1, qid=1, mjd=58050.0)
        params2 = sample_db.get_params(ccdid=2, qid=1, mjd=58050.0)

        # Should be different parameters
        assert not np.allclose(params1, params2)

    def test_get_params_different_mjd(self, sample_db):
        """Test getting parameters for different MJD."""
        params1 = sample_db.get_params(ccdid=1, qid=1, mjd=58050.0)
        params2 = sample_db.get_params(ccdid=1, qid=1, mjd=58150.0)

        # Different time intervals should have different params
        assert not np.allclose(params1, params2)

    def test_get_params_validates_shape(self, sample_db):
        """Test that get_params validates parameter shape."""
        params = sample_db.get_params(ccdid=1, qid=1, mjd=58050.0)

        # Should have correct shape
        assert params.shape == sample_db.model.params_shape

    def test_get_params_no_match(self, sample_db):
        """Test that getting params for non-existent entry raises KeyError."""
        with pytest.raises(KeyError):
            sample_db.get_params(ccdid=999, qid=1, mjd=58050.0)

    def test_get_params_returns_numpy_array(self, sample_db):
        """Test that get_params returns numpy array."""
        params = sample_db.get_params(ccdid=1, qid=1, mjd=58050.0)

        assert isinstance(params, np.ndarray)
        assert params.dtype == np.float64


# ============================================================================
# Test get_eq_func
# ============================================================================


class TestGetEqFunc:
    """Test the get_eq_func method."""

    def test_get_eq_func_returns_jax_func(self, sample_poly_db):
        """Test that get_eq_func with backend='jax' returns JaxEqFunc."""
        grid = np.array([50.0, 100.0, 500.0, 1000.0])
        eq_func = sample_poly_db.get_eq_func(
            ccdid=1,
            qid=1,
            mjd=58050.0,
            ccd_temp=160.0,
            tabulation_grid=grid,
            backend="jax",
        )
        assert isinstance(eq_func, JaxEqFunc)

    def test_get_eq_func_evaluates(self, sample_poly_db):
        """Test that returned eq_func can be evaluated."""
        grid = np.array([50.0, 100.0, 500.0, 1000.0, 5000.0])
        eq_func = sample_poly_db.get_eq_func(
            ccdid=1, qid=1, mjd=58050.0, ccd_temp=160.0, tabulation_grid=grid
        )

        x = np.array([100.0, 500.0, 1000.0])
        y = eq_func(x)

        assert isinstance(y, np.ndarray) or hasattr(y, "__array__")
        assert len(y) == len(x)

    def test_get_eq_func_different_temps(self, sample_poly_db):
        """Test get_eq_func with different temperatures."""
        grid = np.array([50.0, 100.0, 500.0, 1000.0, 5000.0])
        eq_func1 = sample_poly_db.get_eq_func(
            ccdid=1, qid=1, mjd=58050.0, ccd_temp=155.0, tabulation_grid=grid
        )
        eq_func2 = sample_poly_db.get_eq_func(
            ccdid=1, qid=1, mjd=58050.0, ccd_temp=165.0, tabulation_grid=grid
        )

        # Different temps should give different functions
        x = np.array([500.0])
        y1 = eq_func1(x)
        y2 = eq_func2(x)

        assert not np.allclose(y1, y2)

    def test_get_eq_func_custom_grid(self, sample_poly_db):
        """Test get_eq_func with custom tabulation grid."""
        custom_grid = np.array([50.0, 100.0, 500.0, 1000.0])
        eq_func = sample_poly_db.get_eq_func(
            ccdid=1,
            qid=1,
            mjd=58050.0,
            ccd_temp=160.0,
            tabulation_grid=custom_grid,
            backend="jax",
        )
        assert isinstance(eq_func, JaxEqFunc)

    def test_get_eq_func_default_returns_numpy_func(self, sample_poly_db):
        """Test that get_eq_func default (no backend arg) returns NumpyEqFunc."""
        from ztfsensors.pocket.models.base import NumpyEqFunc

        grid = np.array([50.0, 100.0, 500.0, 1000.0])
        eq_func = sample_poly_db.get_eq_func(
            ccdid=1, qid=1, mjd=58050.0, ccd_temp=160.0, tabulation_grid=grid
        )
        assert isinstance(eq_func, NumpyEqFunc)

    def test_get_eq_func_no_match(self, sample_poly_db):
        """Test that get_eq_func raises error for non-existent entry."""
        with pytest.raises(KeyError):
            sample_poly_db.get_eq_func(ccdid=999, qid=1, mjd=58050.0, ccd_temp=160.0)


# ============================================================================
# Test Database Consistency
# ============================================================================


class TestDatabaseConsistency:
    """Test database consistency and data integrity."""

    def test_all_rows_have_params(self, sample_db):
        """Test that all rows have params field."""
        for row in sample_db.df.iter_rows(named=True):
            assert "params" in row
            assert row["params"] is not None

    def test_all_params_have_correct_shape(self, sample_db):
        """Test that all stored params can be reconstructed to correct shape."""
        for row in sample_db.df.iter_rows(named=True):
            params = sample_db.model.validate_params(np.asarray(row["params"]))
            assert params.shape == sample_db.model.params_shape

    def test_mjd_intervals_valid(self, sample_db):
        """Test that all MJD intervals are valid (start < end)."""
        for row in sample_db.df.iter_rows(named=True):
            assert row["mjd_start"] < row["mjd_end"]

    def test_temp_ranges_valid(self, sample_db):
        """Test that all temperature ranges are valid."""
        for row in sample_db.df.iter_rows(named=True):
            assert row["temp_min"] < row["temp_max"]


# ============================================================================
# Test Round-trip Save/Load
# ============================================================================


class TestRoundTrip:
    """Test saving and loading database."""

    def test_roundtrip_preserves_data(self, sample_db_files):
        """Test that opening database preserves data."""
        db = EqFuncDb.open(sample_db_files)

        # Check that we can retrieve data
        params = db.get_params(ccdid=1, qid=1, mjd=58050.0)
        assert params is not None
        assert params.shape == db.model.params_shape

    def test_roundtrip_with_poly_model(self, temp_dir, sample_poly_model):
        """Test round-trip with polynomial model."""
        # Create data - poly model has 1D params, stored flat in the DB.
        records = [
            {
                "ccdid": 1,
                "qid": 1,
                "mjd_start": 58000.0,
                "mjd_end": 58100.0,
                "temp_min": 155.0,
                "temp_max": 165.0,
                "params": np.random.randn(*sample_poly_model.params_shape).tolist(),
            }
        ]
        df = pl.DataFrame(records)

        # Save
        prefix = temp_dir / "poly_db"
        df.write_parquet(prefix.with_suffix(".parquet"))
        header = sample_poly_model.as_dict()
        with prefix.with_suffix(".yaml").open("w") as f:
            yaml.safe_dump(header, f)

        # Load
        db = EqFuncDb.open(prefix)

        assert isinstance(db.model, PolyTempEqModel)
        params = db.get_params(ccdid=1, qid=1, mjd=58050.0)
        assert params.shape == sample_poly_model.params_shape


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_database(self, sample_spline_model):
        """Test database with no records."""
        empty_df = pl.DataFrame(
            {
                "ccdid": [],
                "qid": [],
                "mjd_start": [],
                "mjd_end": [],
                "temp_min": [],
                "temp_max": [],
                "params": [],
            }
        )

        db = EqFuncDb(df=empty_df, model=sample_spline_model)

        with pytest.raises(KeyError):
            db.get_params(ccdid=1, qid=1, mjd=58050.0)

    def test_single_record(self, sample_spline_model):
        """Test database with single record."""
        df = pl.DataFrame(
            [
                {
                    "ccdid": 1,
                    "qid": 1,
                    "mjd_start": 58000.0,
                    "mjd_end": 58100.0,
                    "temp_min": 155.0,
                    "temp_max": 165.0,
                    "params": np.random.randn(
                        *sample_spline_model.params_shape
                    ).tolist(),
                }
            ]
        )

        db = EqFuncDb(df=df, model=sample_spline_model)
        params = db.get_params(ccdid=1, qid=1, mjd=58050.0)

        assert params is not None

    def test_many_ccds(self, sample_spline_model):
        """Test database with many CCDs."""
        records = []
        for ccdid in range(1, 17):  # 16 CCDs
            for qid in range(1, 5):  # 4 quadrants
                records.append(
                    {
                        "ccdid": ccdid,
                        "qid": qid,
                        "mjd_start": 58000.0,
                        "mjd_end": 58100.0,
                        "temp_min": 155.0,
                        "temp_max": 165.0,
                        "params": np.random.randn(
                            *sample_spline_model.params_shape
                        ).tolist(),
                    }
                )

        df = pl.DataFrame(records)
        db = EqFuncDb(df=df, model=sample_spline_model)

        # Should be able to query any CCD/quadrant
        params = db.get_params(ccdid=8, qid=3, mjd=58050.0)
        assert params is not None

    def test_large_mjd_value(self, sample_db):
        """Test with very large MJD values."""
        new_record = {
            "ccdid": 1,
            "qid": 1,
            "mjd_start": 70000.0,
            "mjd_end": 70100.0,
            "temp_min": 155.0,
            "temp_max": 165.0,
            "params": np.random.randn(*sample_db.model.params_shape).tolist(),
        }

        df = sample_db.df.vstack(pl.DataFrame([new_record]))
        db = EqFuncDb(df=df, model=sample_db.model)

        params = db.get_params(ccdid=1, qid=1, mjd=70050.0)
        assert params is not None
