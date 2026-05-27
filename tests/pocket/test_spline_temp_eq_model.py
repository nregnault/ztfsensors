"""
Tests for SplineTempEqModel.

This module tests the spline-based temperature-dependent equilibrium model,
including initialization, design matrix construction, parameter handling,
flattening/unflattening, and serialization.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from ztfsensors.pocket.models.spline_temp_eq_model import SplineTempEqModel

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_spline_model():
    """Create a SplineTempEqModel with default parameters."""
    return SplineTempEqModel()


@pytest.fixture
def custom_spline_model():
    """Create a SplineTempEqModel with custom parameters."""
    basis_grid = np.array([50.0, 100.0, 500.0, 1000.0, 5000.0])
    return SplineTempEqModel(
        basis_grid=basis_grid,
        temp_deg=3,
        temp_ref=155.0,
        temp_scale=2.0,
        basis_order=3,
    )


@pytest.fixture
def simple_spline_model():
    """Create a simple SplineTempEqModel for basic tests."""
    basis_grid = np.array([100.0, 500.0, 1000.0])
    return SplineTempEqModel(
        basis_grid=basis_grid,
        temp_deg=2,
        temp_ref=160.0,
        temp_scale=1.0,
        basis_order=2,
    )


# ============================================================================
# Test Initialization
# ============================================================================


class TestInitialization:
    """Test model initialization and attributes."""

    def test_default_initialization(self, default_spline_model):
        """Test that default parameters are set correctly."""
        model = default_spline_model
        assert model.temp_deg == 5
        assert model.temp_ref == 160.0
        assert model.temp_scale == 1.0
        assert model.basis_order == 4
        # Default grid is geomspace(50, 10000, 10)
        assert len(model.basis_grid) == 10
        assert model.basis_grid[0] == 50.0
        assert model.basis_grid[-1] == 10000.0

    def test_custom_initialization(self, custom_spline_model):
        """Test initialization with custom parameters."""
        model = custom_spline_model
        assert model.temp_deg == 3
        assert model.temp_ref == 155.0
        assert model.temp_scale == 2.0
        assert model.basis_order == 3
        assert len(model.basis_grid) == 5
        np.testing.assert_array_equal(
            model.basis_grid, [50.0, 100.0, 500.0, 1000.0, 5000.0]
        )

    def test_basis_grid_as_array(self):
        """Test that basis_grid is stored as numpy array."""
        grid = [100.0, 200.0, 300.0]
        model = SplineTempEqModel(basis_grid=grid)
        assert isinstance(model.basis_grid, np.ndarray)
        np.testing.assert_array_equal(model.basis_grid, grid)

    def test_temp_ref_as_float(self):
        """Test that temp_ref is stored as float."""
        model = SplineTempEqModel(temp_ref=160)
        assert isinstance(model.temp_ref, float)
        assert model.temp_ref == 160.0

    def test_temp_scale_as_float(self):
        """Test that temp_scale is stored as float."""
        model = SplineTempEqModel(temp_scale=5)
        assert isinstance(model.temp_scale, float)
        assert model.temp_scale == 5.0

    def test_temp_deg_as_int(self):
        """Test that temp_deg is stored as int."""
        model = SplineTempEqModel(temp_deg=3.0)
        assert isinstance(model.temp_deg, int)
        assert model.temp_deg == 3

    def test_basis_order_as_int(self):
        """Test that basis_order is stored as int."""
        model = SplineTempEqModel(basis_order=3.0)
        assert isinstance(model.basis_order, int)
        assert model.basis_order == 3

    def test_basis_size_is_int(self, default_spline_model):
        """Test that basis_size is stored as int."""
        assert isinstance(default_spline_model.basis_size, int)

    def test_zero_temp_scale_raises_error(self):
        """Test that zero temp_scale raises ValueError."""
        with pytest.raises(ValueError, match="temp_scale must be non-zero"):
            SplineTempEqModel(temp_scale=0.0)

    def test_basis_size_validation(self):
        """Test that provided basis_size is validated against computed size."""
        grid = np.geomspace(50.0, 1000.0, 5)
        model = SplineTempEqModel(basis_grid=grid, basis_order=4)

        # This should work - correct basis_size
        correct_size = model.basis_size
        model2 = SplineTempEqModel(
            basis_grid=grid, basis_order=4, basis_size=correct_size
        )
        assert model2.basis_size == correct_size

        # This should fail - wrong basis_size
        with pytest.raises(ValueError, match="Provided basis_size"):
            SplineTempEqModel(basis_grid=grid, basis_order=4, basis_size=999)

    def test_model_name_and_version(self, default_spline_model):
        """Test that model has correct name and version."""
        assert default_spline_model.MODEL_NAME == "spline_temp_eq"
        assert default_spline_model.MODEL_VERSION == "v1"

    def test_header_fields(self, default_spline_model):
        """Test that HEADER_FIELDS is correctly defined."""
        expected_fields = (
            "basis_grid",
            "temp_deg",
            "temp_ref",
            "temp_scale",
            "basis_size",
            "basis_order",
        )
        assert default_spline_model.HEADER_FIELDS == expected_fields


# ============================================================================
# Test default_basis_grid
# ============================================================================


class TestDefaultBasisGrid:
    """Test the default basis grid generation."""

    def test_default_basis_grid_returns_array(self):
        """Test that default_basis_grid returns numpy array."""
        model = SplineTempEqModel()
        grid = model.default_basis_grid()
        assert isinstance(grid, np.ndarray)

    def test_default_basis_grid_values(self):
        """Test that default grid has expected values."""
        model = SplineTempEqModel()
        grid = model.default_basis_grid()

        assert len(grid) == 10
        assert grid[0] == 50.0
        assert grid[-1] == 10000.0
        # Should be geometric progression
        ratios = grid[1:] / grid[:-1]
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-10)

    def test_default_basis_grid_is_sorted(self):
        """Test that default grid is sorted."""
        model = SplineTempEqModel()
        grid = model.default_basis_grid()
        assert np.all(grid[1:] > grid[:-1])


# ============================================================================
# Test params_shape
# ============================================================================


class TestParamsShape:
    """Test the params_shape property."""

    def test_params_shape_default(self, default_spline_model):
        """Test params_shape with default parameters."""
        model = default_spline_model
        # 1D: basis_size * (temp_deg + 1) = basis_size * 6
        assert model.params_shape == (model.n_model_coeffs,)

    def test_params_shape_custom(self, custom_spline_model):
        """Test params_shape with custom parameters."""
        model = custom_spline_model
        # temp_deg=3  →  n_model_coeffs = basis_size * 4
        assert model.params_shape == (model.basis_size * 4,)

    def test_params_shape_simple(self, simple_spline_model):
        """Test params_shape with simple model."""
        model = simple_spline_model
        # temp_deg=2  →  n_model_coeffs = basis_size * 3
        assert model.params_shape == (model.basis_size * 3,)

    def test_params_shape_is_1d(self, default_spline_model):
        """Test that params_shape is a 1-D tuple."""
        shape = default_spline_model.params_shape
        assert len(shape) == 1
        assert isinstance(shape[0], int)

    def test_params_shape_temp_deg_zero(self):
        """Test params_shape with temp_deg=0."""
        model = SplineTempEqModel(temp_deg=0)
        # 1 temperature coefficient per basis function
        assert model.params_shape == (model.basis_size,)


# ============================================================================
# Test n_model_coeffs
# ============================================================================


class TestNModelCoeffs:
    """Test the n_model_coeffs property."""

    def test_n_model_coeffs_matches_shape(self, default_spline_model):
        """Test that n_model_coeffs equals the length of params_shape."""
        model = default_spline_model
        assert model.n_model_coeffs == model.params_shape[0]

    def test_n_model_coeffs_custom(self, custom_spline_model):
        """Test n_model_coeffs with custom model."""
        model = custom_spline_model
        expected = model.basis_size * (model.temp_deg + 1)
        assert model.n_model_coeffs == expected

    def test_n_model_coeffs_is_int(self, default_spline_model):
        """Test that n_model_coeffs is an integer."""
        assert isinstance(default_spline_model.n_model_coeffs, int)


# ============================================================================
# Test build_design_matrix
# ============================================================================


class TestBuildDesignMatrix:
    """Test the design matrix construction."""

    def test_design_matrix_shape(self, simple_spline_model):
        """Test that design matrix has correct shape."""
        n = 50
        x = np.linspace(10.0, 500.0, n)
        temp = np.linspace(155.0, 165.0, n)

        J = simple_spline_model.build_design_matrix(x, temp)

        assert J.shape == (n, simple_spline_model.n_model_coeffs)

    def test_design_matrix_type(self, simple_spline_model):
        """Test that design matrix is CSR sparse matrix."""
        x = np.array([100.0, 200.0])
        temp = np.array([160.0, 165.0])

        J = simple_spline_model.build_design_matrix(x, temp)

        assert isinstance(J, sparse.csr_matrix)

    def test_design_matrix_basic_evaluation(self, simple_spline_model):
        """Test design matrix construction with basic inputs."""
        x = np.array([100.0, 500.0, 1000.0])
        temp = np.array([160.0, 160.0, 160.0])

        J = simple_spline_model.build_design_matrix(x, temp)

        assert J.shape[0] == len(x)
        assert not np.any(np.isnan(J.data))
        assert not np.any(np.isinf(J.data))

    def test_design_matrix_temperature_variation(self, simple_spline_model):
        """Test that temperature variation affects design matrix."""
        x = np.array([100.0, 100.0])
        temp = np.array([150.0, 170.0])

        J = simple_spline_model.build_design_matrix(x, temp)
        dense_J = J.toarray()

        # Same x but different temps should give different rows
        assert not np.allclose(dense_J[0], dense_J[1])

    def test_design_matrix_x_variation(self, simple_spline_model):
        """Test that x variation affects design matrix."""
        x = np.array([100.0, 500.0])
        temp = np.array([160.0, 160.0])

        J = simple_spline_model.build_design_matrix(x, temp)
        dense_J = J.toarray()

        # Different x but same temp should give different rows
        assert not np.allclose(dense_J[0], dense_J[1])

    def test_design_matrix_mismatched_shapes(self, default_spline_model):
        """Test that mismatched x and temp shapes raise ValueError."""
        x = np.array([10.0, 100.0, 500.0])
        temp = np.array([160.0, 165.0])  # Different length

        with pytest.raises(ValueError, match="x and ccd_temp must have same shape"):
            default_spline_model.build_design_matrix(x, temp)

    def test_design_matrix_2d_arrays(self, default_spline_model):
        """Test design matrix with 2D arrays."""
        x = np.array([[10.0, 100.0], [200.0, 500.0]])
        temp = np.array([[160.0, 161.0], [162.0, 163.0]])

        # Flatten the arrays first as BSpline expects 1D input
        x_flat = x.flatten()
        temp_flat = temp.flatten()

        J = default_spline_model.build_design_matrix(x_flat, temp_flat)

        # Should have 4 rows (flattened from 2x2)
        assert J.shape[0] == 4

    def test_design_matrix_scalar_inputs(self, default_spline_model):
        """Test design matrix with scalar inputs."""
        x = np.array([100.0])
        temp = np.array([160.0])

        J = default_spline_model.build_design_matrix(x, temp)

        assert J.shape[0] == 1

    def test_design_matrix_temp_at_ref(self, custom_spline_model):
        """Test design matrix when temp equals temp_ref."""
        # When temp = temp_ref, dt = 0
        x = np.array([100.0, 500.0])
        temp = np.array([155.0, 155.0])  # temp_ref = 155.0

        J = custom_spline_model.build_design_matrix(x, temp)
        dense_J = J.toarray()

        # Only the dt^0 terms should be non-zero
        # These are the last basis_size columns
        n_basis = custom_spline_model.basis_size
        temp_deg = custom_spline_model.temp_deg

        # First columns correspond to high temp powers, should be zero
        for i in range(temp_deg):
            assert np.allclose(dense_J[:, i * n_basis : (i + 1) * n_basis], 0.0)

    def test_design_matrix_large_dataset(self, default_spline_model):
        """Test design matrix with large dataset."""
        n = 1000
        x = np.random.uniform(50.0, 5000.0, n)
        temp = np.random.uniform(155.0, 165.0, n)

        J = default_spline_model.build_design_matrix(x, temp)

        assert J.shape[0] == n
        assert not np.any(np.isnan(J.data))


# ============================================================================
# Test Model Registration
# ============================================================================


class TestModelRegistration:
    """Test that the model is properly registered with the factory."""

    def test_model_registered(self):
        """Test that SplineTempEqModel is registered in the factory."""
        from ztfsensors.pocket.models.factory import EquilibriumModelFactory

        assert ("spline_temp_eq", "v1") in EquilibriumModelFactory._registry

    def test_factory_creates_model(self):
        """Test that factory can create SplineTempEqModel."""
        from ztfsensors.pocket.models.factory import EquilibriumModelFactory

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

        model = EquilibriumModelFactory.from_header(header)
        assert isinstance(model, SplineTempEqModel)
        assert model.temp_deg == 3


# ============================================================================
# Test Serialization
# ============================================================================


class TestSerialization:
    """Test model serialization and deserialization."""

    def test_as_dict(self, default_spline_model):
        """Test that as_dict returns correct dictionary."""
        model_dict = default_spline_model.as_dict()

        assert model_dict["model_name"] == "spline_temp_eq"
        assert model_dict["model_version"] == "v1"
        assert "basis_grid" in model_dict
        assert model_dict["temp_deg"] == 5
        assert model_dict["temp_ref"] == 160.0
        assert model_dict["temp_scale"] == 1.0
        assert model_dict["basis_order"] == 4
        assert "basis_size" in model_dict
        assert "param_layout" not in model_dict

    def test_as_dict_custom(self, custom_spline_model):
        """Test as_dict with custom parameters."""
        model_dict = custom_spline_model.as_dict()

        assert model_dict["temp_deg"] == 3
        assert model_dict["temp_ref"] == 155.0
        assert model_dict["temp_scale"] == 2.0
        assert model_dict["basis_order"] == 3
        assert len(model_dict["basis_grid"]) == 5

    def test_roundtrip_serialization(self, custom_spline_model):
        """Test that model can be serialized and deserialized."""
        from ztfsensors.pocket.models.factory import EquilibriumModelFactory

        # Serialize
        model_dict = custom_spline_model.as_dict()

        # Deserialize
        restored_model = EquilibriumModelFactory.from_header(model_dict)

        # Check attributes match
        assert restored_model.temp_deg == custom_spline_model.temp_deg
        assert restored_model.temp_ref == custom_spline_model.temp_ref
        assert restored_model.temp_scale == custom_spline_model.temp_scale
        assert restored_model.basis_order == custom_spline_model.basis_order
        assert restored_model.basis_size == custom_spline_model.basis_size
        np.testing.assert_array_equal(
            restored_model.basis_grid, custom_spline_model.basis_grid
        )

    def test_basis_grid_serialization(self, custom_spline_model):
        """Test that basis_grid is correctly serialized."""
        model_dict = custom_spline_model.as_dict()

        # basis_grid should be present and convertible to array
        assert "basis_grid" in model_dict
        grid = np.array(model_dict["basis_grid"])
        np.testing.assert_array_equal(grid, custom_spline_model.basis_grid)


# ============================================================================
# Test Parameter Validation
# ============================================================================


class TestParameterValidation:
    """Test parameter validation."""

    def test_validate_params_correct_shape(self, default_spline_model):
        """Test that validate_params accepts correct shape."""
        params = np.random.randn(*default_spline_model.params_shape)
        validated = default_spline_model.validate_params(params)
        np.testing.assert_array_equal(validated, params)

    def test_validate_params_wrong_shape(self, default_spline_model):
        """Test that validate_params rejects wrong shape."""
        wrong_params = np.random.randn(10, 10)  # Wrong shape

        with pytest.raises(ValueError):
            default_spline_model.validate_params(wrong_params)

    def test_validate_params_wrong_length(self, default_spline_model):
        """Test that validate_params rejects a 1D array of wrong length."""
        wrong_params = np.random.randn(50)  # Wrong length (not n_model_coeffs)

        with pytest.raises(ValueError):
            default_spline_model.validate_params(wrong_params)

    def test_validate_params_non_numeric(self, default_spline_model):
        """Test that validate_params rejects non-numeric params."""
        n = default_spline_model.n_model_coeffs
        params = np.array(["a"] * n)  # String array, correct length but wrong dtype

        with pytest.raises(TypeError):
            default_spline_model.validate_params(params)


# ============================================================================
# Test Equilibrium Function Evaluation
# ============================================================================


class TestEquilibriumEvaluation:
    """Test model evaluation."""

    def test_evaluate_basic(self, simple_spline_model):
        """Test basic model evaluation."""
        params = np.ones(simple_spline_model.params_shape)

        x = np.array([50.0, 100.0, 200.0])
        temp = np.array([160.0, 160.0, 160.0])

        J = simple_spline_model.build_design_matrix(x, temp)
        y = J @ params

        assert y.shape == x.shape
        assert not np.any(np.isnan(y))
        assert not np.any(np.isinf(y))

    def test_evaluate_matches_design_matrix(self, simple_spline_model):
        """Test that evaluate gives same result as design matrix multiplication."""
        params = np.random.randn(*simple_spline_model.params_shape)

        x = np.array([50.0, 100.0, 200.0, 500.0])
        temp = np.array([155.0, 160.0, 165.0, 170.0])

        # Via design matrix
        J = simple_spline_model.build_design_matrix(x, temp)
        y_matrix = J @ params

        # Both methods should give same result
        assert y_matrix.shape == x.shape
        assert not np.any(np.isnan(y_matrix))

    def test_evaluate_temperature_dependence(self, default_spline_model):
        """Test that evaluation depends on temperature."""
        params = np.random.randn(*default_spline_model.params_shape)

        x = np.array([100.0, 100.0])
        temp = np.array([150.0, 170.0])

        # Via design matrix
        J = default_spline_model.build_design_matrix(x, temp)
        y = J @ params

        # Same x but different temps should give different results
        assert y[0] != y[1]

    def test_evaluate_vectorized(self, default_spline_model):
        """Test vectorized evaluation."""
        params = np.random.randn(*default_spline_model.params_shape)

        n = 100
        x = np.random.uniform(50.0, 5000.0, n)
        temp = np.random.uniform(155.0, 165.0, n)

        # Via design matrix
        J = default_spline_model.build_design_matrix(x, temp)
        y = J @ params

        assert y.shape == (n,)
        assert not np.any(np.isnan(y))


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_very_small_x(self, default_spline_model):
        """Test with very small x values."""
        x = np.array([0.0, 1e-6, 1e-3])
        temp = np.array([160.0, 160.0, 160.0])

        J = default_spline_model.build_design_matrix(x, temp)
        assert J.shape[0] == 3

    def test_very_large_x(self, default_spline_model):
        """Test with very large x values."""
        x = np.array([10000.0, 50000.0, 100000.0])
        temp = np.array([160.0, 160.0, 160.0])

        J = default_spline_model.build_design_matrix(x, temp)
        assert J.shape[0] == 3

    def test_extreme_temperatures(self, default_spline_model):
        """Test with extreme temperature values."""
        x = np.array([100.0, 200.0])
        temp = np.array([100.0, 200.0])  # Outside typical range

        J = default_spline_model.build_design_matrix(x, temp)
        assert not np.any(np.isnan(J.data))
        assert not np.any(np.isinf(J.data))

    def test_single_point(self, default_spline_model):
        """Test with single data point."""
        x = np.array([100.0])
        temp = np.array([160.0])

        J = default_spline_model.build_design_matrix(x, temp)
        assert J.shape == (1, default_spline_model.n_model_coeffs)

    def test_temp_deg_zero(self):
        """Test model with temp_deg=0 (no temperature dependence)."""
        model = SplineTempEqModel(temp_deg=0)

        x = np.array([100.0, 200.0])
        temp = np.array([155.0, 165.0])

        J = model.build_design_matrix(x, temp)
        # Should have basis_size columns only
        assert J.shape[1] == model.basis_size


# ============================================================================
# Test Inheritance
# ============================================================================


class TestInheritance:
    """Test that SplineTempEqModel properly inherits from BaseEquilibriumModel."""

    def test_is_base_model(self, default_spline_model):
        """Test that model is instance of BaseEquilibriumModel."""
        from ztfsensors.pocket.models.base import BaseEquilibriumModel

        assert isinstance(default_spline_model, BaseEquilibriumModel)

    def test_has_rescale_temp(self, default_spline_model):
        """Test that model inherits rescale_temp method."""
        temp = np.array([155.0, 160.0, 165.0])
        dt = default_spline_model.rescale_temp(temp)

        expected = (temp - 160.0) / 1.0
        np.testing.assert_allclose(dt, expected)

    def test_rescale_temp_custom_scale(self, custom_spline_model):
        """Test rescale_temp with custom scale."""
        temp = np.array([155.0, 157.0, 159.0])
        dt = custom_spline_model.rescale_temp(temp)

        expected = (temp - 155.0) / 2.0
        np.testing.assert_allclose(dt, expected)


# ============================================================================
# Test Integration with BSpline
# ============================================================================


class TestBSplineIntegration:
    """Test integration with bbf.bspline.BSpline."""

    def test_basis_is_bspline(self, default_spline_model):
        """Test that basis attribute is a BSpline instance."""
        from bbf.bspline import BSpline

        assert isinstance(default_spline_model.basis, BSpline)

    def test_basis_has_correct_order(self, custom_spline_model):
        """Test that BSpline has correct order."""
        assert custom_spline_model.basis.order == 3

    def test_basis_size_matches_len(self, default_spline_model):
        """Test that basis_size matches len(basis)."""
        assert default_spline_model.basis_size == len(default_spline_model.basis)

    def test_basis_evaluates(self, simple_spline_model):
        """Test that basis can be evaluated."""
        x = np.array([100.0, 500.0, 1000.0])
        B = simple_spline_model.basis.eval(x)

        assert B.shape[0] == len(x)
        assert B.shape[1] == simple_spline_model.basis_size


# ============================================================================
# Test Comparison with Different Configurations
# ============================================================================


class TestDifferentConfigurations:
    """Test model behavior with different configurations."""

    def test_higher_basis_order(self):
        """Test model with higher spline order."""
        model = SplineTempEqModel(basis_order=5, temp_deg=2)

        x = np.array([100.0, 500.0])
        temp = np.array([160.0, 165.0])

        J = model.build_design_matrix(x, temp)
        assert J.shape[0] == 2

    def test_lower_basis_order(self):
        """Test model with lower spline order."""
        model = SplineTempEqModel(basis_order=2, temp_deg=2)

        x = np.array([100.0, 500.0])
        temp = np.array([160.0, 165.0])

        J = model.build_design_matrix(x, temp)
        assert J.shape[0] == 2

    def test_dense_basis_grid(self):
        """Test model with dense basis grid."""
        dense_grid = np.geomspace(50.0, 5000.0, 50)
        model = SplineTempEqModel(basis_grid=dense_grid)

        assert model.basis_size >= len(dense_grid)

    def test_sparse_basis_grid(self):
        """Test model with sparse basis grid."""
        sparse_grid = np.array([50.0, 1000.0, 5000.0])
        model = SplineTempEqModel(basis_grid=sparse_grid)

        x = np.array([100.0, 500.0])
        temp = np.array([160.0, 165.0])

        J = model.build_design_matrix(x, temp)
        assert J.shape[0] == 2

    def test_high_temp_deg(self):
        """Test model with high temperature degree."""
        model = SplineTempEqModel(temp_deg=10)

        assert model.params_shape == (model.n_model_coeffs,)
        assert model.n_model_coeffs == model.basis_size * 11
