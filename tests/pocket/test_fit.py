"""
Tests for fit module.

This module tests the fitting utilities for equilibrium models, including
FitRecord, FitResults, FitDiagnostics, and the fit_eq_model function.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from ztfsensors.pocket.fit import (
    FitDiagnostics,
    FitRecord,
    FitResults,
    fit_eq_model,
)
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
def simple_spline_model():
    """Create a simple spline model for testing."""
    return SplineTempEqModel(
        basis_grid=np.geomspace(50.0, 5000.0, 8),
        temp_deg=3,
        temp_ref=160.0,
        temp_scale=1.0,
        basis_order=4,
    )


@pytest.fixture
def simple_poly_model():
    """Create a simple poly model for testing."""
    return PolyTempEqModel(
        p_deg=[2, 2],
        q_deg=[2],
        x_knot=200.0,
        temp_ref=160.0,
        temp_scale=1.0,
    )


@pytest.fixture
def sample_fit_record(simple_poly_model):
    """Create a sample FitRecord."""
    params = np.random.randn(*simple_poly_model.params_shape)
    return FitRecord(
        ccdid=1,
        qid=1,
        mjd_start=58000.0,
        mjd_end=58100.0,
        temp_min=155.0,
        temp_max=165.0,
        params=params,
    )


@pytest.fixture
def synthetic_fit_data():
    """Generate synthetic data for fitting tests."""
    np.random.seed(42)
    n = 200

    # Sky levels
    skylev = np.random.uniform(50.0, 5000.0, n)

    # Temperatures
    cryotemp = np.random.uniform(155.0, 165.0, n)

    # Synthetic overscan signal: simple model
    u = np.log1p(skylev)
    overscan_sum = 50.0 * u + 5.0 * (cryotemp - 160.0) + np.random.normal(0, 2.0, n)

    # MJD
    mjd = np.random.uniform(58000.0, 58100.0, n)

    # Create DataFrame
    df = pl.DataFrame(
        {
            "ccdid": np.ones(n, dtype=int),
            "qid": np.ones(n, dtype=int),
            "skylev": skylev,
            "overscan_sum": overscan_sum,
            "cryotemp": cryotemp,
            "mjd": mjd,
        }
    )

    return df


# ============================================================================
# Test FitRecord
# ============================================================================


class TestFitRecord:
    """Test FitRecord dataclass."""

    def test_init(self, simple_poly_model):
        """Test FitRecord initialization."""
        params = np.random.randn(*simple_poly_model.params_shape)

        record = FitRecord(
            ccdid=1,
            qid=2,
            mjd_start=58000.0,
            mjd_end=58100.0,
            temp_min=155.0,
            temp_max=165.0,
            params=params,
        )

        assert record.ccdid == 1
        assert record.qid == 2
        assert record.mjd_start == 58000.0
        assert record.mjd_end == 58100.0
        assert record.temp_min == 155.0
        assert record.temp_max == 165.0
        np.testing.assert_array_equal(record.params, params)

    def test_frozen(self, sample_fit_record):
        """Test that FitRecord is frozen (immutable)."""
        with pytest.raises(AttributeError):
            sample_fit_record.ccdid = 999

    def test_validate_correct(self, sample_fit_record, simple_poly_model):
        """Test validation with correct record."""
        validated_params = sample_fit_record.validate(simple_poly_model)

        assert validated_params.shape == simple_poly_model.params_shape
        np.testing.assert_array_equal(validated_params, sample_fit_record.params)

    def test_validate_invalid_mjd_interval(self, simple_poly_model):
        """Test validation fails with invalid MJD interval."""
        params = np.random.randn(*simple_poly_model.params_shape)
        record = FitRecord(
            ccdid=1,
            qid=1,
            mjd_start=58100.0,
            mjd_end=58000.0,  # End before start!
            temp_min=155.0,
            temp_max=165.0,
            params=params,
        )

        with pytest.raises(ValueError, match="Invalid MJD interval"):
            record.validate(simple_poly_model)

    def test_validate_invalid_temp_interval(self, simple_poly_model):
        """Test validation fails with invalid temperature interval."""
        params = np.random.randn(*simple_poly_model.params_shape)
        record = FitRecord(
            ccdid=1,
            qid=1,
            mjd_start=58000.0,
            mjd_end=58100.0,
            temp_min=155.0,
            temp_max=155.0,  # Same as min!
            params=params,
        )

        with pytest.raises(ValueError, match="Invalid temperature interval"):
            record.validate(simple_poly_model)

    def test_validate_wrong_params_shape(self, simple_poly_model):
        """Test validation fails with wrong parameter shape."""
        wrong_params = np.random.randn(10, 10)  # Wrong shape
        record = FitRecord(
            ccdid=1,
            qid=1,
            mjd_start=58000.0,
            mjd_end=58100.0,
            temp_min=155.0,
            temp_max=165.0,
            params=wrong_params,
        )

        with pytest.raises(ValueError):
            record.validate(simple_poly_model)


# ============================================================================
# Test FitResults
# ============================================================================


class TestFitResults:
    """Test FitResults dataclass."""

    def test_init_empty(self, simple_poly_model):
        """Test FitResults initialization with no records."""
        results = FitResults(model=simple_poly_model)

        assert results.model is simple_poly_model
        assert results.records == []

    def test_init_with_records(self, simple_poly_model, sample_fit_record):
        """Test FitResults initialization with records."""
        results = FitResults(model=simple_poly_model, records=[sample_fit_record])

        assert len(results.records) == 1
        assert results.records[0] is sample_fit_record

    def test_append(self, simple_poly_model, sample_fit_record):
        """Test appending a record."""
        results = FitResults(model=simple_poly_model)
        results.append(sample_fit_record)

        assert len(results.records) == 1
        assert results.records[0] is sample_fit_record

    def test_append_validates_record(self, simple_poly_model):
        """Test that append validates the record."""
        results = FitResults(model=simple_poly_model)

        # Invalid record (bad MJD interval)
        params = np.random.randn(*simple_poly_model.params_shape)
        bad_record = FitRecord(
            ccdid=1,
            qid=1,
            mjd_start=58100.0,
            mjd_end=58000.0,  # Invalid!
            temp_min=155.0,
            temp_max=165.0,
            params=params,
        )

        with pytest.raises(ValueError):
            results.append(bad_record)

    def test_to_dataframe_empty(self, simple_poly_model):
        """Test converting empty results to DataFrame."""
        results = FitResults(model=simple_poly_model)
        df = results.to_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert df.height == 0
        assert "ccdid" in df.columns
        assert "params" in df.columns

    def test_to_dataframe_with_records(self, simple_poly_model):
        """Test converting results with records to DataFrame."""
        params1 = np.random.randn(*simple_poly_model.params_shape)
        params2 = np.random.randn(*simple_poly_model.params_shape)

        record1 = FitRecord(
            ccdid=1,
            qid=1,
            mjd_start=58000.0,
            mjd_end=58100.0,
            temp_min=155.0,
            temp_max=165.0,
            params=params1,
        )
        record2 = FitRecord(
            ccdid=1,
            qid=2,
            mjd_start=58000.0,
            mjd_end=58100.0,
            temp_min=156.0,
            temp_max=166.0,
            params=params2,
        )

        results = FitResults(model=simple_poly_model, records=[record1, record2])
        df = results.to_dataframe()

        assert df.height == 2
        assert df["ccdid"].to_list() == [1, 1]
        assert df["qid"].to_list() == [1, 2]

    def test_to_dataframe_params_as_nested_list(self, simple_poly_model):
        """Test that params are stored as nested lists in DataFrame."""
        params = np.random.randn(*simple_poly_model.params_shape)
        record = FitRecord(
            ccdid=1,
            qid=1,
            mjd_start=58000.0,
            mjd_end=58100.0,
            temp_min=155.0,
            temp_max=165.0,
            params=params,
        )

        results = FitResults(model=simple_poly_model, records=[record])
        df = results.to_dataframe()

        # Params should be a list (1D for poly model)
        params_from_df = df["params"].to_list()[0]
        assert isinstance(params_from_df, list)
        assert len(params_from_df) == simple_poly_model.params_shape[0]

    def test_validate_no_duplicates(self, simple_poly_model):
        """Test that validate passes with no duplicate records."""
        params1 = np.random.randn(*simple_poly_model.params_shape)
        params2 = np.random.randn(*simple_poly_model.params_shape)

        record1 = FitRecord(
            ccdid=1,
            qid=1,
            mjd_start=58000.0,
            mjd_end=58100.0,
            temp_min=155.0,
            temp_max=165.0,
            params=params1,
        )
        record2 = FitRecord(
            ccdid=1,
            qid=1,
            mjd_start=58100.0,
            mjd_end=58200.0,
            temp_min=155.0,
            temp_max=165.0,
            params=params2,
        )

        results = FitResults(model=simple_poly_model, records=[record1, record2])
        results.validate()  # Should not raise

    def test_validate_detects_duplicates(self, simple_poly_model):
        """Test that validate detects duplicate records."""
        params = np.random.randn(*simple_poly_model.params_shape)

        record1 = FitRecord(
            ccdid=1,
            qid=1,
            mjd_start=58000.0,
            mjd_end=58100.0,
            temp_min=155.0,
            temp_max=165.0,
            params=params,
        )
        record2 = FitRecord(
            ccdid=1,
            qid=1,
            mjd_start=58000.0,
            mjd_end=58100.0,  # Same!
            temp_min=155.0,
            temp_max=165.0,
            params=params,
        )

        results = FitResults(model=simple_poly_model, records=[record1, record2])

        with pytest.raises(ValueError, match="Duplicate exact blocks found"):
            results.validate()

    def test_validate_detects_overlapping_intervals(self, simple_poly_model):
        """Test that validate detects overlapping MJD intervals."""
        params1 = np.random.randn(*simple_poly_model.params_shape)
        params2 = np.random.randn(*simple_poly_model.params_shape)

        record1 = FitRecord(
            ccdid=1,
            qid=1,
            mjd_start=58000.0,
            mjd_end=58150.0,
            temp_min=155.0,
            temp_max=165.0,
            params=params1,
        )
        record2 = FitRecord(
            ccdid=1,
            qid=1,
            mjd_start=58100.0,
            mjd_end=58200.0,  # Overlaps!
            temp_min=155.0,
            temp_max=165.0,
            params=params2,
        )

        results = FitResults(model=simple_poly_model, records=[record1, record2])

        with pytest.raises(ValueError, match="Overlapping MJD intervals"):
            results.validate()

    def test_validate_allows_touching_intervals(self, simple_poly_model):
        """Test that validate allows touching (but not overlapping) intervals."""
        params1 = np.random.randn(*simple_poly_model.params_shape)
        params2 = np.random.randn(*simple_poly_model.params_shape)

        record1 = FitRecord(
            ccdid=1,
            qid=1,
            mjd_start=58000.0,
            mjd_end=58100.0,
            temp_min=155.0,
            temp_max=165.0,
            params=params1,
        )
        record2 = FitRecord(
            ccdid=1,
            qid=1,
            mjd_start=58100.0,
            mjd_end=58200.0,  # Touches at 58100
            temp_min=155.0,
            temp_max=165.0,
            params=params2,
        )

        results = FitResults(model=simple_poly_model, records=[record1, record2])
        results.validate()  # Should not raise

    def test_save(self, temp_dir, simple_poly_model, sample_fit_record):
        """Test saving results to files."""
        results = FitResults(model=simple_poly_model, records=[sample_fit_record])
        prefix = temp_dir / "test_results"

        results.save(prefix)

        # Check files exist
        assert prefix.with_suffix(".parquet").exists()
        assert prefix.with_suffix(".yaml").exists()

    def test_save_creates_directory(
        self, temp_dir, simple_poly_model, sample_fit_record
    ):
        """Test that save creates parent directories."""
        results = FitResults(model=simple_poly_model, records=[sample_fit_record])
        prefix = temp_dir / "subdir" / "test_results"

        results.save(prefix)

        assert prefix.with_suffix(".parquet").exists()
        assert prefix.with_suffix(".yaml").exists()

    def test_save_yaml_contains_model_info(
        self, temp_dir, simple_poly_model, sample_fit_record
    ):
        """Test that saved YAML contains model information."""
        results = FitResults(model=simple_poly_model, records=[sample_fit_record])
        prefix = temp_dir / "test_results"

        results.save(prefix)

        # Read YAML
        with prefix.with_suffix(".yaml").open("r") as f:
            header = yaml.safe_load(f)

        assert header["model_name"] == "poly_temp_eq"
        assert header["model_version"] == "v1"
        assert "p_deg" in header
        assert "q_deg" in header

    def test_save_validates_before_saving(self, temp_dir, simple_poly_model):
        """Test that save validates before writing."""
        params = np.random.randn(*simple_poly_model.params_shape)

        # Create duplicate records
        record1 = FitRecord(
            ccdid=1,
            qid=1,
            mjd_start=58000.0,
            mjd_end=58100.0,
            temp_min=155.0,
            temp_max=165.0,
            params=params,
        )
        record2 = FitRecord(
            ccdid=1,
            qid=1,
            mjd_start=58000.0,
            mjd_end=58100.0,
            temp_min=155.0,
            temp_max=165.0,
            params=params,
        )

        results = FitResults(model=simple_poly_model, records=[record1, record2])
        prefix = temp_dir / "test_results"

        with pytest.raises(ValueError, match="Duplicate exact blocks found"):
            results.save(prefix)


# ============================================================================
# Test FitDiagnostics
# ============================================================================


class TestFitDiagnostics:
    """Test FitDiagnostics dataclass."""

    def test_from_fit_data(self, simple_poly_model, sample_fit_record):
        """Test creating diagnostics from fit data."""
        n = 50
        xx = np.random.uniform(50.0, 2000.0, n)
        yy = np.random.uniform(0.0, 300.0, n)
        temp = np.random.uniform(155.0, 165.0, n)
        mjd = np.random.uniform(58000.0, 58100.0, n)

        diag = FitDiagnostics.from_fit_data(
            model=simple_poly_model,
            record=sample_fit_record,
            xx=xx,
            yy=yy,
            temp=temp,
            mjd=mjd,
        )

        assert diag.model is simple_poly_model
        assert diag.record is sample_fit_record
        np.testing.assert_array_equal(diag.xx, xx)
        np.testing.assert_array_equal(diag.yy, yy)
        np.testing.assert_array_equal(diag.temp, temp)
        np.testing.assert_array_equal(diag.mjd, mjd)
        assert diag.yhat.shape == yy.shape
        assert diag.resid.shape == yy.shape

    def test_from_fit_data_computes_residuals(
        self, simple_poly_model, sample_fit_record
    ):
        """Test that residuals are computed correctly."""
        n = 50
        xx = np.random.uniform(50.0, 2000.0, n)
        yy = np.random.uniform(0.0, 300.0, n)
        temp = np.random.uniform(155.0, 165.0, n)
        mjd = np.random.uniform(58000.0, 58100.0, n)

        diag = FitDiagnostics.from_fit_data(
            model=simple_poly_model,
            record=sample_fit_record,
            xx=xx,
            yy=yy,
            temp=temp,
            mjd=mjd,
        )

        # Check that resid = yy - yhat
        expected_resid = yy - diag.yhat
        np.testing.assert_allclose(diag.resid, expected_resid)

    def test_from_fit_data_with_bads(self, simple_poly_model, sample_fit_record):
        """Test creating diagnostics with bad point flags."""
        n = 50
        xx = np.random.uniform(50.0, 2000.0, n)
        yy = np.random.uniform(0.0, 300.0, n)
        temp = np.random.uniform(155.0, 165.0, n)
        mjd = np.random.uniform(58000.0, 58100.0, n)
        bads = np.random.rand(n) > 0.9  # ~10% bad

        diag = FitDiagnostics.from_fit_data(
            model=simple_poly_model,
            record=sample_fit_record,
            xx=xx,
            yy=yy,
            temp=temp,
            mjd=mjd,
            bads=bads,
        )

        np.testing.assert_array_equal(diag.bads_, bads)

    def test_from_fit_data_mismatched_shapes(
        self, simple_poly_model, sample_fit_record
    ):
        """Test that mismatched array shapes raise error."""
        xx = np.random.uniform(50.0, 2000.0, 50)
        yy = np.random.uniform(0.0, 300.0, 60)  # Different length!
        temp = np.random.uniform(155.0, 165.0, 50)
        mjd = np.random.uniform(58000.0, 58100.0, 50)

        with pytest.raises(ValueError, match="must have the same shape"):
            FitDiagnostics.from_fit_data(
                model=simple_poly_model,
                record=sample_fit_record,
                xx=xx,
                yy=yy,
                temp=temp,
                mjd=mjd,
            )

    def test_bads_property_with_bads(self, simple_poly_model, sample_fit_record):
        """Test bads property when bads_ is set."""
        n = 50
        xx = np.random.uniform(50.0, 2000.0, n)
        yy = np.random.uniform(0.0, 300.0, n)
        temp = np.random.uniform(155.0, 165.0, n)
        mjd = np.random.uniform(58000.0, 58100.0, n)
        bads = np.random.rand(n) > 0.9

        diag = FitDiagnostics.from_fit_data(
            model=simple_poly_model,
            record=sample_fit_record,
            xx=xx,
            yy=yy,
            temp=temp,
            mjd=mjd,
            bads=bads,
        )

        np.testing.assert_array_equal(diag.bads, bads)

    def test_bads_property_without_bads(self, simple_poly_model, sample_fit_record):
        """Test bads property when bads_ is None."""
        n = 50
        xx = np.random.uniform(50.0, 2000.0, n)
        yy = np.random.uniform(0.0, 300.0, n)
        temp = np.random.uniform(155.0, 165.0, n)
        mjd = np.random.uniform(58000.0, 58100.0, n)

        diag = FitDiagnostics.from_fit_data(
            model=simple_poly_model,
            record=sample_fit_record,
            xx=xx,
            yy=yy,
            temp=temp,
            mjd=mjd,
        )

        # Should return all True (all good points)
        assert diag.bads.shape == (n,)
        assert diag.bads.all()


# ============================================================================
# Test fit_eq_model
# ============================================================================


class TestFitEqModel:
    """Test the fit_eq_model function."""

    def test_fit_basic(self, synthetic_fit_data, simple_poly_model):
        """Test basic fitting functionality."""
        record, diag = fit_eq_model(
            df=synthetic_fit_data,
            ccdid=1,
            qid=1,
            eq_model=simple_poly_model,
            robust_fit=False,
        )

        assert isinstance(record, FitRecord)
        assert isinstance(diag, FitDiagnostics)
        assert record.ccdid == 1
        assert record.qid == 1

    def test_fit_returns_correct_shapes(self, synthetic_fit_data, simple_poly_model):
        """Test that fit returns correct parameter shapes."""
        record, diag = fit_eq_model(
            df=synthetic_fit_data,
            ccdid=1,
            qid=1,
            eq_model=simple_poly_model,
            robust_fit=False,
        )

        assert record.params.shape == simple_poly_model.params_shape

    def test_fit_mjd_interval(self, synthetic_fit_data, simple_poly_model):
        """Test that MJD interval is set correctly."""
        record, diag = fit_eq_model(
            df=synthetic_fit_data,
            ccdid=1,
            qid=1,
            eq_model=simple_poly_model,
            robust_fit=False,
        )

        mjd_min = synthetic_fit_data["mjd"].min()
        mjd_max = synthetic_fit_data["mjd"].max()

        assert record.mjd_start == mjd_min
        assert record.mjd_end == mjd_max + 1.0

    def test_fit_temp_range(self, synthetic_fit_data, simple_poly_model):
        """Test that temperature range is set correctly."""
        record, diag = fit_eq_model(
            df=synthetic_fit_data,
            ccdid=1,
            qid=1,
            eq_model=simple_poly_model,
            robust_fit=False,
        )

        temp_min = synthetic_fit_data["cryotemp"].min()
        temp_max = synthetic_fit_data["cryotemp"].max()

        assert record.temp_min == temp_min
        assert record.temp_max == temp_max

    def test_fit_with_sky_filtering(self, synthetic_fit_data, simple_poly_model):
        """Test that sky level filtering works."""
        record, diag = fit_eq_model(
            df=synthetic_fit_data,
            ccdid=1,
            qid=1,
            eq_model=simple_poly_model,
            sky_min=100.0,
            sky_max=1000.0,
            robust_fit=False,
        )

        # Diagnostic data should only include filtered sky levels
        assert np.all(diag.xx >= 100.0)
        assert np.all(diag.xx <= 1000.0)

    def test_fit_with_y_filtering(self, synthetic_fit_data, simple_poly_model):
        """Test that overscan signal filtering works."""
        record, diag = fit_eq_model(
            df=synthetic_fit_data,
            ccdid=1,
            qid=1,
            eq_model=simple_poly_model,
            y_min=50.0,
            y_max=500.0,
            robust_fit=False,
        )

        # Diagnostic data should only include filtered y values
        assert np.all(diag.yy >= 50.0)
        assert np.all(diag.yy <= 500.0)

    def test_fit_robust(self, synthetic_fit_data, simple_poly_model):
        """Test robust fitting."""
        record, diag = fit_eq_model(
            df=synthetic_fit_data,
            ccdid=1,
            qid=1,
            eq_model=simple_poly_model,
            robust_fit=True,
        )

        assert isinstance(record, FitRecord)
        # With robust fit, bads_ should be set
        assert diag.bads_ is not None

    def test_fit_non_robust(self, synthetic_fit_data, simple_poly_model):
        """Test non-robust fitting."""
        record, diag = fit_eq_model(
            df=synthetic_fit_data,
            ccdid=1,
            qid=1,
            eq_model=simple_poly_model,
            robust_fit=False,
        )

        assert isinstance(record, FitRecord)
        # With non-robust fit, bads_ should be None
        assert diag.bads_ is None

    def test_fit_different_ccd(self):
        """Test fitting different CCDs."""
        np.random.seed(123)
        n = 100

        # Create data for two CCDs
        df = pl.DataFrame(
            {
                "ccdid": [1] * (n // 2) + [2] * (n // 2),
                "qid": [1] * n,
                "skylev": np.random.uniform(50.0, 2000.0, n),
                "overscan_sum": np.random.uniform(50.0, 300.0, n),
                "cryotemp": np.random.uniform(155.0, 165.0, n),
                "mjd": np.random.uniform(58000.0, 58100.0, n),
            }
        )

        model = PolyTempEqModel(p_deg=[1, 1], q_deg=[1])

        record1, _ = fit_eq_model(df, ccdid=1, qid=1, eq_model=model, robust_fit=False)
        record2, _ = fit_eq_model(df, ccdid=2, qid=1, eq_model=model, robust_fit=False)

        assert record1.ccdid == 1
        assert record2.ccdid == 2
        # Parameters should be different for different CCDs
        assert not np.allclose(record1.params, record2.params)

    def test_fit_with_custom_beta(self, synthetic_fit_data, simple_poly_model):
        """Test fitting with custom beta regularization."""
        record, diag = fit_eq_model(
            df=synthetic_fit_data,
            ccdid=1,
            qid=1,
            eq_model=simple_poly_model,
            beta=1e-5,
            robust_fit=False,
        )

        assert isinstance(record, FitRecord)

    def test_fit_no_data(self, simple_poly_model):
        """Test fitting with no matching data."""
        # Empty DataFrame
        df = pl.DataFrame(
            {
                "ccdid": [],
                "qid": [],
                "skylev": [],
                "overscan_sum": [],
                "cryotemp": [],
                "mjd": [],
            }
        )

        # Should raise an error or handle gracefully
        # This depends on implementation - it might fail in the linear solver
        with pytest.raises(Exception):
            fit_eq_model(
                df, ccdid=1, qid=1, eq_model=simple_poly_model, robust_fit=False
            )

    def test_fit_diagnostics_match_data(self, synthetic_fit_data, simple_poly_model):
        """Test that diagnostics match the filtered data."""
        record, diag = fit_eq_model(
            df=synthetic_fit_data,
            ccdid=1,
            qid=1,
            eq_model=simple_poly_model,
            sky_min=100.0,
            sky_max=1000.0,
            robust_fit=False,
        )

        # Filter the data manually
        filtered = synthetic_fit_data.filter(
            (pl.col("ccdid") == 1)
            & (pl.col("qid") == 1)
            & pl.col("skylev").is_between(100.0, 1000.0)
        )

        # Diagnostics should have same length as filtered data
        assert len(diag.xx) == filtered.height

    def test_fit_residuals_make_sense(self, synthetic_fit_data, simple_poly_model):
        """Test that fit residuals are reasonable."""
        record, diag = fit_eq_model(
            df=synthetic_fit_data,
            ccdid=1,
            qid=1,
            eq_model=simple_poly_model,
            robust_fit=False,
        )

        # Residuals should have similar magnitude to input noise (which is ~2.0)
        # Mean residual should be close to zero
        assert abs(np.mean(diag.resid)) < 1.0
        # Std of residuals should be reasonable
        assert np.std(diag.resid) < 10.0


# ============================================================================
# Test Integration
# ============================================================================


class TestIntegration:
    """Test integration between components."""

    def test_full_workflow(self, temp_dir, synthetic_fit_data, simple_poly_model):
        """Test complete workflow: fit -> save -> load."""
        # Fit
        record, diag = fit_eq_model(
            df=synthetic_fit_data,
            ccdid=1,
            qid=1,
            eq_model=simple_poly_model,
            robust_fit=True,
        )

        # Create results and save
        results = FitResults(model=simple_poly_model)
        results.append(record)

        prefix = temp_dir / "workflow_test"
        results.save(prefix)

        # Verify files exist
        assert prefix.with_suffix(".parquet").exists()
        assert prefix.with_suffix(".yaml").exists()

        # Load and verify
        df_loaded = pl.read_parquet(prefix.with_suffix(".parquet"))
        assert df_loaded.height == 1
        assert df_loaded["ccdid"][0] == 1

    def test_multiple_fits(self, synthetic_fit_data, simple_poly_model):
        """Test fitting multiple CCDs/quadrants."""
        results = FitResults(model=simple_poly_model)

        # Fit different quadrants
        for qid in [1, 2]:
            # Add some data for qid=2
            if qid == 2:
                # Create new data with qid=2
                df = synthetic_fit_data.with_columns(
                    pl.lit(2, dtype=pl.Int64).alias("qid")
                )
            else:
                df = synthetic_fit_data

            record, _ = fit_eq_model(
                df=df,
                ccdid=1,
                qid=qid,
                eq_model=simple_poly_model,
                robust_fit=False,
            )
            results.append(record)

        assert len(results.records) == 2
        results.validate()  # Should not raise
