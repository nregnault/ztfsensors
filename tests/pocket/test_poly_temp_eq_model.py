"""
Tests for PolyTempEqModel.

This module tests the polynomial temperature-dependent equilibrium model,
including initialization, design matrix construction, parameter handling,
and serialization.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from ztfsensors.pocket.models.poly_temp_eq_model import PolyTempEqModel

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_poly_model():
    """Create a PolyTempEqModel with default parameters."""
    return PolyTempEqModel()


@pytest.fixture
def custom_poly_model():
    """Create a PolyTempEqModel with custom parameters."""
    return PolyTempEqModel(
        p_deg=[1, 2, None],  # Include None to test skipping
        q_deg=[2, 3],
        x_knot=150.0,
        temp_ref=155.0,
        temp_scale=2.0,
    )


@pytest.fixture
def simple_poly_model():
    """Create a simple PolyTempEqModel for basic tests."""
    return PolyTempEqModel(
        p_deg=[1, 1],
        q_deg=[1],
        x_knot=100.0,
        temp_ref=160.0,
        temp_scale=1.0,
    )


# ============================================================================
# Test Initialization
# ============================================================================


class TestInitialization:
    """Test model initialization and attributes."""

    def test_default_initialization(self, default_poly_model):
        """Test that default parameters are set correctly."""
        model = default_poly_model
        assert model.p_deg == [2, 2, 2]
        assert model.q_deg == [3, 3]
        assert model.x_knot == 200.0
        assert model.temp_ref == 160.0
        assert model.temp_scale == 1.0
        assert model.u_knot == np.log(1.0 + 200.0)

    def test_custom_initialization(self, custom_poly_model):
        """Test initialization with custom parameters."""
        model = custom_poly_model
        assert model.p_deg == [1, 2, None]
        assert model.q_deg == [2, 3]
        assert model.x_knot == 150.0
        assert model.temp_ref == 155.0
        assert model.temp_scale == 2.0
        assert model.u_knot == np.log(1.0 + 150.0)

    def test_u_knot_calculation(self):
        """Test that u_knot is correctly computed from x_knot."""
        x_knot = 300.0
        model = PolyTempEqModel(x_knot=x_knot)
        expected_u_knot = np.log(1.0 + x_knot)
        np.testing.assert_allclose(model.u_knot, expected_u_knot)

    def test_temp_ref_as_float(self):
        """Test that temp_ref is stored as float."""
        model = PolyTempEqModel(temp_ref=160)
        assert isinstance(model.temp_ref, float)
        assert model.temp_ref == 160.0

    def test_temp_scale_as_float(self):
        """Test that temp_scale is stored as float."""
        model = PolyTempEqModel(temp_scale=5)
        assert isinstance(model.temp_scale, float)
        assert model.temp_scale == 5.0

    def test_model_name_and_version(self, default_poly_model):
        """Test that model has correct name and version."""
        assert default_poly_model.MODEL_NAME == "poly_temp_eq"
        assert default_poly_model.MODEL_VERSION == "v1"

    def test_header_fields(self, default_poly_model):
        """Test that HEADER_FIELDS is correctly defined."""
        expected_fields = ("p_deg", "q_deg", "x_knot", "temp_ref", "temp_scale")
        assert default_poly_model.HEADER_FIELDS == expected_fields


# ============================================================================
# Test params_shape
# ============================================================================


class TestParamsShape:
    """Test the params_shape property."""

    def test_params_shape_default(self, default_poly_model):
        """Test params_shape with default parameters."""
        # p_deg=[2,2,2] -> 3+3+3=9 params
        # q_deg=[3,3] -> 4+4=8 params
        # Total: 17 params
        expected_shape = (9 + 8,)
        assert default_poly_model.params_shape == expected_shape

    def test_params_shape_custom(self, custom_poly_model):
        """Test params_shape with custom parameters including None."""
        # p_deg=[1,2,None] -> 2+3+0=5 params
        # q_deg=[2,3] -> 3+4=7 params
        # Total: 12 params
        expected_shape = (5 + 7,)
        assert custom_poly_model.params_shape == expected_shape

    def test_params_shape_simple(self, simple_poly_model):
        """Test params_shape with simple model."""
        # p_deg=[1,1] -> 2+2=4 params
        # q_deg=[1] -> 2 params
        # Total: 6 params
        expected_shape = (6,)
        assert simple_poly_model.params_shape == expected_shape

    def test_params_shape_all_none_p_deg(self):
        """Test params_shape when all p_deg are None."""
        model = PolyTempEqModel(p_deg=[None, None], q_deg=[2])
        # p_deg=[None,None] -> 0 params
        # q_deg=[2] -> 3 params
        expected_shape = (3,)
        assert model.params_shape == expected_shape

    def test_params_shape_empty_lists(self):
        """Test params_shape with empty degree lists."""
        model = PolyTempEqModel(p_deg=[], q_deg=[1])
        expected_shape = (2,)  # Only q_deg contributes
        assert model.params_shape == expected_shape


# ============================================================================
# Test build_design_matrix
# ============================================================================


class TestBuildDesignMatrix:
    """Test the design matrix construction."""

    def test_design_matrix_shape(self, simple_poly_model):
        """Test that design matrix has correct shape."""
        n = 50
        x = np.linspace(10.0, 500.0, n)
        temp = np.linspace(155.0, 165.0, n)

        J = simple_poly_model.build_design_matrix(x, temp)

        assert isinstance(J, sparse.coo_matrix)
        assert J.shape == (n, simple_poly_model.params_shape[0])

    def test_design_matrix_positive_x(self, default_poly_model):
        """Test design matrix construction with positive x values."""
        x = np.array([10.0, 100.0, 500.0])
        temp = np.array([160.0, 160.0, 160.0])

        J = default_poly_model.build_design_matrix(x, temp)

        assert J.shape[0] == len(x)
        assert not np.any(np.isnan(J.data))
        assert not np.any(np.isinf(J.data))

    def test_design_matrix_x_at_knot(self, default_poly_model):
        """Test design matrix when x equals x_knot."""
        x = np.array([200.0])  # x_knot = 200.0 by default
        temp = np.array([160.0])

        J = default_poly_model.build_design_matrix(x, temp)

        # Should work without errors
        assert J.shape[0] == 1

    def test_design_matrix_x_above_knot(self, default_poly_model):
        """Test design matrix when x > x_knot."""
        x = np.array([500.0, 1000.0])  # x_knot = 200.0
        temp = np.array([160.0, 165.0])

        J = default_poly_model.build_design_matrix(x, temp)

        # du should be positive for these x values
        assert J.shape[0] == 2
        # Check that q_deg terms contribute (non-zero in last columns)
        dense_J = J.toarray()
        assert np.any(dense_J[:, -4:] != 0)  # Last 4+4 columns from q_deg

    def test_design_matrix_x_below_knot(self, default_poly_model):
        """Test design matrix when x < x_knot."""
        x = np.array([50.0, 100.0])  # x_knot = 200.0
        temp = np.array([160.0, 165.0])

        J = default_poly_model.build_design_matrix(x, temp)

        # du should be zero for these x values
        assert J.shape[0] == 2
        dense_J = J.toarray()
        # Last columns (from q_deg terms with du) should be zero
        np.testing.assert_allclose(dense_J[:, -8:], 0.0)

    def test_design_matrix_temperature_variation(self, simple_poly_model):
        """Test that temperature variation affects design matrix."""
        x = np.array([100.0, 100.0])
        temp = np.array([150.0, 170.0])

        J = simple_poly_model.build_design_matrix(x, temp)
        dense_J = J.toarray()

        # Same x but different temps should give different rows
        assert not np.allclose(dense_J[0], dense_J[1])

    def test_design_matrix_with_none_in_p_deg(self, custom_poly_model):
        """Test design matrix when p_deg contains None."""
        x = np.array([100.0, 200.0])
        temp = np.array([160.0, 165.0])

        J = custom_poly_model.build_design_matrix(x, temp)

        # Should work and have correct shape
        assert J.shape[0] == 2
        assert J.shape[1] == custom_poly_model.params_shape[0]

    def test_design_matrix_invalid_x(self, default_poly_model):
        """Test that x <= -1 raises ValueError."""
        x = np.array([-2.0, 0.0, 10.0])
        temp = np.array([160.0, 160.0, 160.0])

        with pytest.raises(ValueError, match="x must satisfy x > -1"):
            default_poly_model.build_design_matrix(x, temp)

    def test_design_matrix_x_at_boundary(self, default_poly_model):
        """Test that x slightly above -1 works."""
        x = np.array([-0.9, 0.0, 10.0])
        temp = np.array([160.0, 160.0, 160.0])

        J = default_poly_model.build_design_matrix(x, temp)
        assert J.shape[0] == 3

    def test_design_matrix_mismatched_shapes(self, default_poly_model):
        """Test that mismatched x and temp shapes raise ValueError."""
        x = np.array([10.0, 100.0, 500.0])
        temp = np.array([160.0, 165.0])  # Different length

        with pytest.raises(ValueError, match="x and ccd_temp must have same shape"):
            default_poly_model.build_design_matrix(x, temp)

    def test_design_matrix_2d_arrays(self, default_poly_model):
        """Test design matrix with 2D arrays."""
        x = np.array([[10.0, 100.0], [200.0, 500.0]])
        temp = np.array([[160.0, 161.0], [162.0, 163.0]])

        # Flatten the arrays first as the model expects 1D input
        x_flat = x.flatten()
        temp_flat = temp.flatten()

        J = default_poly_model.build_design_matrix(x_flat, temp_flat)

        # Should have 4 rows (flattened from 2x2)
        assert J.shape[0] == 4

    def test_design_matrix_scalar_inputs(self, default_poly_model):
        """Test design matrix with scalar inputs."""
        x = np.array([100.0])
        temp = np.array([160.0])

        J = default_poly_model.build_design_matrix(x, temp)

        assert J.shape[0] == 1

    def test_design_matrix_sparse_format(self, default_poly_model):
        """Test that returned matrix is COO sparse format."""
        x = np.array([100.0, 200.0])
        temp = np.array([160.0, 165.0])

        J = default_poly_model.build_design_matrix(x, temp)

        assert isinstance(J, sparse.coo_matrix)
        assert hasattr(J, "row")
        assert hasattr(J, "col")
        assert hasattr(J, "data")


# ============================================================================
# Test Model Registration
# ============================================================================


class TestModelRegistration:
    """Test that the model is properly registered with the factory."""

    def test_model_registered(self):
        """Test that PolyTempEqModel is registered in the factory."""
        from ztfsensors.pocket.models.factory import EquilibriumModelFactory

        assert ("poly_temp_eq", "v1") in EquilibriumModelFactory._registry

    def test_factory_creates_model(self):
        """Test that factory can create PolyTempEqModel."""
        from ztfsensors.pocket.models.factory import EquilibriumModelFactory

        header = {
            "model_name": "poly_temp_eq",
            "model_version": "v1",
            "p_deg": [2, 2, 2],
            "q_deg": [3, 3],
            "x_knot": 200.0,
            "temp_ref": 160.0,
            "temp_scale": 1.0,
        }

        model = EquilibriumModelFactory.from_header(header)
        assert isinstance(model, PolyTempEqModel)
        assert model.p_deg == [2, 2, 2]


# ============================================================================
# Test Serialization
# ============================================================================


class TestSerialization:
    """Test model serialization and deserialization."""

    def test_as_dict(self, default_poly_model):
        """Test that as_dict returns correct dictionary."""
        model_dict = default_poly_model.as_dict()

        assert model_dict["model_name"] == "poly_temp_eq"
        assert model_dict["model_version"] == "v1"
        assert model_dict["p_deg"] == [2, 2, 2]
        assert model_dict["q_deg"] == [3, 3]
        assert model_dict["x_knot"] == 200.0
        assert model_dict["temp_ref"] == 160.0
        assert model_dict["temp_scale"] == 1.0

    def test_as_dict_custom(self, custom_poly_model):
        """Test as_dict with custom parameters."""
        model_dict = custom_poly_model.as_dict()

        assert model_dict["p_deg"] == [1, 2, None]
        assert model_dict["q_deg"] == [2, 3]
        assert model_dict["x_knot"] == 150.0
        assert model_dict["temp_ref"] == 155.0
        assert model_dict["temp_scale"] == 2.0

    def test_roundtrip_serialization(self, custom_poly_model):
        """Test that model can be serialized and deserialized."""
        from ztfsensors.pocket.models.factory import EquilibriumModelFactory

        # Serialize
        model_dict = custom_poly_model.as_dict()

        # Deserialize
        restored_model = EquilibriumModelFactory.from_header(model_dict)

        # Check attributes match
        assert restored_model.p_deg == custom_poly_model.p_deg
        assert restored_model.q_deg == custom_poly_model.q_deg
        assert restored_model.x_knot == custom_poly_model.x_knot
        assert restored_model.temp_ref == custom_poly_model.temp_ref
        assert restored_model.temp_scale == custom_poly_model.temp_scale
        assert restored_model.u_knot == custom_poly_model.u_knot


# ============================================================================
# Test Parameter Validation
# ============================================================================


class TestParameterValidation:
    """Test parameter validation."""

    def test_validate_params_correct_shape(self, default_poly_model):
        """Test that validate_params accepts correct shape."""
        params = np.random.randn(*default_poly_model.params_shape)
        validated = default_poly_model.validate_params(params)
        np.testing.assert_array_equal(validated, params)

    def test_validate_params_wrong_shape(self, default_poly_model):
        """Test that validate_params rejects wrong shape."""
        wrong_params = np.random.randn(10)  # Wrong size

        with pytest.raises(ValueError):
            default_poly_model.validate_params(wrong_params)

    def test_validate_params_non_numeric(self, default_poly_model):
        """Test that validate_params rejects non-numeric params."""
        params = np.array(["a", "b"] * 8 + ["c"])  # String array

        with pytest.raises(TypeError):
            default_poly_model.validate_params(params)


# ============================================================================
# Test Equilibrium Function Evaluation
# ============================================================================


class TestEquilibriumEvaluation:
    """Test model evaluation."""

    def test_evaluate_basic(self, simple_poly_model):
        """Test basic model evaluation."""
        # Create simple params
        params = np.ones(simple_poly_model.params_shape)

        x = np.array([50.0, 100.0, 200.0])
        temp = np.array([160.0, 160.0, 160.0])

        y = simple_poly_model.evaluate(x, temp, params)

        assert y.shape == x.shape
        assert not np.any(np.isnan(y))
        assert not np.any(np.isinf(y))

    def test_evaluate_matches_design_matrix(self, simple_poly_model):
        """Test that evaluate gives same result as design matrix multiplication."""
        params = np.random.randn(*simple_poly_model.params_shape)

        x = np.array([50.0, 100.0, 200.0, 500.0])
        temp = np.array([155.0, 160.0, 165.0, 170.0])

        # Direct evaluation
        y_direct = simple_poly_model.evaluate(x, temp, params)

        # Via design matrix
        J = simple_poly_model.build_design_matrix(x, temp)
        y_matrix = J @ params

        np.testing.assert_allclose(y_direct, y_matrix)

    def test_evaluate_temperature_dependence(self, default_poly_model):
        """Test that evaluation depends on temperature."""
        params = np.random.randn(*default_poly_model.params_shape)

        x = np.array([100.0, 100.0])
        temp = np.array([150.0, 170.0])

        y = default_poly_model.evaluate(x, temp, params)

        # Same x but different temps should give different results
        assert y[0] != y[1]


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_very_small_x(self, default_poly_model):
        """Test with very small but valid x values."""
        x = np.array([0.0, 1e-6, 1e-3])
        temp = np.array([160.0, 160.0, 160.0])

        J = default_poly_model.build_design_matrix(x, temp)
        assert J.shape[0] == 3

    def test_very_large_x(self, default_poly_model):
        """Test with very large x values."""
        x = np.array([10000.0, 50000.0, 100000.0])
        temp = np.array([160.0, 160.0, 160.0])

        J = default_poly_model.build_design_matrix(x, temp)
        assert J.shape[0] == 3
        assert not np.any(np.isnan(J.data))

    def test_extreme_temperatures(self, default_poly_model):
        """Test with extreme temperature values."""
        x = np.array([100.0, 200.0])
        temp = np.array([100.0, 200.0])  # Outside typical range

        J = default_poly_model.build_design_matrix(x, temp)
        assert not np.any(np.isnan(J.data))
        assert not np.any(np.isinf(J.data))

    def test_single_point(self, default_poly_model):
        """Test with single data point."""
        x = np.array([100.0])
        temp = np.array([160.0])

        J = default_poly_model.build_design_matrix(x, temp)
        assert J.shape == (1, default_poly_model.params_shape[0])

    def test_empty_q_deg(self):
        """Test model with empty q_deg."""
        model = PolyTempEqModel(p_deg=[2, 2], q_deg=[])

        x = np.array([100.0, 200.0])
        temp = np.array([160.0, 165.0])

        J = model.build_design_matrix(x, temp)
        # Should only have p_deg contributions
        assert J.shape[1] == 3 + 3  # Two terms with degree 2 each


# ============================================================================
# Test Inheritance
# ============================================================================


class TestInheritance:
    """Test that PolyTempEqModel properly inherits from BaseEquilibriumModel."""

    def test_is_base_model(self, default_poly_model):
        """Test that model is instance of BaseEquilibriumModel."""
        from ztfsensors.pocket.models.base import BaseEquilibriumModel

        assert isinstance(default_poly_model, BaseEquilibriumModel)

    def test_has_rescale_temp(self, default_poly_model):
        """Test that model inherits rescale_temp method."""
        temp = np.array([155.0, 160.0, 165.0])
        dt = default_poly_model.rescale_temp(temp)

        expected = (temp - 160.0) / 1.0
        np.testing.assert_allclose(dt, expected)

    def test_rescale_temp_custom_scale(self, custom_poly_model):
        """Test rescale_temp with custom scale."""
        temp = np.array([155.0, 157.0, 159.0])
        dt = custom_poly_model.rescale_temp(temp)

        expected = (temp - 155.0) / 2.0
        np.testing.assert_allclose(dt, expected)
