"""
Tests for base equilibrium model classes.

This module tests the base classes JaxEqFunc and BaseEquilibriumModel,
which provide the foundation for all equilibrium models.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import sparse

from ztfsensors.pocket.models.base import BaseEquilibriumModel, JaxEqFunc

# ============================================================================
# Test fixtures and helper classes
# ============================================================================


class DummyEquilibriumModel(BaseEquilibriumModel):
    """
    Minimal concrete implementation of BaseEquilibriumModel for testing.

    This model implements a simple linear relationship:
        y = a * x + b * temp

    where params = [a, b].
    """

    MODEL_NAME = "dummy"
    MODEL_VERSION = "v1"
    HEADER_FIELDS = ("temp_ref", "temp_scale")

    def __init__(self, temp_ref: float = 160.0, temp_scale: float = 1.0):
        self.temp_ref = float(temp_ref)
        self.temp_scale = float(temp_scale)

    @property
    def params_shape(self) -> tuple[int]:
        return (2,)

    def build_design_matrix(self, x, ccd_temp):
        """Build design matrix: [x, temp]."""
        x = np.asarray(x)
        ccd_temp = np.asarray(ccd_temp)

        if x.shape != ccd_temp.shape:
            raise ValueError("x and ccd_temp must have same shape")

        dt = self.rescale_temp(ccd_temp)
        J = np.column_stack([x, dt])
        return sparse.csr_matrix(J)


class DummyModelNoTemp(BaseEquilibriumModel):
    """Dummy model without temperature attributes."""

    MODEL_NAME = "dummy_notemp"
    MODEL_VERSION = "v1"
    HEADER_FIELDS = ("scale",)

    def __init__(self, scale: float = 1.0):
        self.scale = float(scale)

    @property
    def params_shape(self) -> tuple[int]:
        return (1,)

    def build_design_matrix(self, x, ccd_temp):
        x = np.asarray(x)
        return sparse.csr_matrix(np.atleast_2d(x).T * self.scale)


@pytest.fixture
def dummy_model():
    """Create a dummy equilibrium model for testing."""
    return DummyEquilibriumModel(temp_ref=160.0, temp_scale=10.0)


@pytest.fixture
def dummy_model_notemp():
    """Create a dummy model without temperature attributes."""
    return DummyModelNoTemp(scale=2.0)


@pytest.fixture
def simple_grid():
    """Create simple x and y grids for JaxEqFunc testing."""
    x_grid = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    # y_grid has maximum at the end to avoid clipping during interpolation tests
    y_grid = jnp.array([0.0, 1.0, 2.0, 2.5, 3.0], dtype=jnp.float32)
    return x_grid, y_grid


# ============================================================================
# Tests for JaxEqFunc
# ============================================================================


class TestJaxEqFunc:
    """Tests for the JaxEqFunc class."""

    def test_initialization(self, simple_grid):
        """Test that JaxEqFunc initializes correctly."""
        x_grid, y_grid = simple_grid
        eq_func = JaxEqFunc(x_grid, y_grid)

        assert jnp.array_equal(eq_func.x_grid, x_grid)
        assert jnp.array_equal(eq_func.y_grid, y_grid)
        # x_max should be at index 4 where y_grid is maximum (3.0)
        assert eq_func.x_max == x_grid[4]

    def test_interpolation_at_grid_points(self, simple_grid):
        """Test that function evaluates correctly at grid points."""
        x_grid, y_grid = simple_grid
        eq_func = JaxEqFunc(x_grid, y_grid)

        for i, (x, y_expected) in enumerate(zip(x_grid, y_grid)):
            y = eq_func(x)
            assert jnp.isclose(y, y_expected), f"Failed at grid point {i}"

    def test_interpolation_between_points(self, simple_grid):
        """Test linear interpolation between grid points."""
        x_grid, y_grid = simple_grid
        eq_func = JaxEqFunc(x_grid, y_grid)

        # Test midpoint between x=1 and x=2
        # y should be (1.0 + 2.0) / 2 = 1.5
        y = eq_func(1.5)
        assert jnp.isclose(y, 1.5)

    def test_clipping_below_xmin(self, simple_grid):
        """Test that values below x_min are clipped."""
        x_grid, y_grid = simple_grid
        eq_func = JaxEqFunc(x_grid, y_grid)

        # x < x_min should give y at x_min
        y = eq_func(-1.0)
        assert jnp.isclose(y, y_grid[0])

    def test_clipping_above_xmax(self, simple_grid):
        """Test that values above x_max are clipped."""
        x_grid, y_grid = simple_grid
        eq_func = JaxEqFunc(x_grid, y_grid)

        # x > x_max should give y at x_max
        # x_max is at index 4 (x=4.0, y=3.0)
        y = eq_func(10.0)
        assert jnp.isclose(y, y_grid[4])

    def test_vectorized_evaluation(self, simple_grid):
        """Test that function works with array inputs."""
        x_grid, y_grid = simple_grid
        eq_func = JaxEqFunc(x_grid, y_grid)

        x_test = jnp.array([0.0, 1.0, 2.0])
        y_test = eq_func(x_test)

        expected = jnp.array([0.0, 1.0, 2.0])
        assert jnp.allclose(y_test, expected)

    def test_different_input_types(self, simple_grid):
        """Test that function accepts different input types."""
        x_grid, y_grid = simple_grid
        eq_func = JaxEqFunc(x_grid, y_grid)

        # Scalar
        y_scalar = eq_func(1.0)
        assert jnp.ndim(y_scalar) == 0

        # List
        y_list = eq_func([1.0, 2.0])
        assert jnp.ndim(y_list) == 1

        # NumPy array
        y_numpy = eq_func(np.array([1.0, 2.0]))
        assert jnp.ndim(y_numpy) == 1


# ============================================================================
# Tests for BaseEquilibriumModel
# ============================================================================


class TestBaseEquilibriumModel:
    """Tests for the BaseEquilibriumModel abstract class."""

    def test_model_has_required_attributes(self, dummy_model):
        """Test that model has required class attributes."""
        assert hasattr(dummy_model, "MODEL_NAME")
        assert hasattr(dummy_model, "MODEL_VERSION")
        assert hasattr(dummy_model, "HEADER_FIELDS")

    def test_params_shape_property(self, dummy_model):
        """Test that params_shape returns correct shape."""
        shape = dummy_model.params_shape
        assert shape == (2,)

    def test_validate_params_correct_shape(self, dummy_model):
        """Test validate_params with correct shape."""
        params = np.array([1.0, 2.0])
        validated = dummy_model.validate_params(params)

        assert isinstance(validated, np.ndarray)
        assert validated.shape == (2,)
        assert np.allclose(validated, params)

    def test_validate_params_wrong_shape(self, dummy_model):
        """Test that validate_params raises error on wrong shape."""
        params = np.array([1.0, 2.0, 3.0])  # Wrong: should be (2,)

        with pytest.raises(ValueError, match="expected .* got"):
            dummy_model.validate_params(params)

    def test_validate_params_non_numeric(self, dummy_model):
        """Test that validate_params raises error on non-numeric data."""
        params = np.array(["a", "b"])

        with pytest.raises(TypeError, match="must be numeric"):
            dummy_model.validate_params(params)

    def test_flat_to_params_default(self, dummy_model):
        """Test default flat_to_params returns array as-is."""
        params = np.array([1.0, 2.0])
        result = dummy_model.flat_to_params(params)

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, params)

    def test_rescale_temp_with_attributes(self, dummy_model):
        """Test rescale_temp when model has temp_ref and temp_scale."""
        temp = np.array([160.0, 170.0, 150.0])
        rescaled = dummy_model.rescale_temp(temp)

        expected = (temp - 160.0) / 10.0
        assert np.allclose(rescaled, expected)

    def test_rescale_temp_without_attributes(self, dummy_model_notemp):
        """Test rescale_temp when model lacks temp attributes."""
        temp = np.array([160.0, 170.0, 150.0])
        rescaled = dummy_model_notemp.rescale_temp(temp)

        # Should return temp unchanged
        assert np.array_equal(rescaled, temp)

    def test_build_design_matrix(self, dummy_model):
        """Test build_design_matrix creates correct matrix."""
        x = np.array([100.0, 200.0, 300.0])
        temp = np.array([160.0, 165.0, 170.0])

        J = dummy_model.build_design_matrix(x, temp)

        assert J.shape == (3, 2)
        # First column should be x
        assert np.allclose(J[:, 0].toarray().ravel(), x)
        # Second column should be rescaled temp
        expected_temp = (temp - 160.0) / 10.0
        assert np.allclose(J[:, 1].toarray().ravel(), expected_temp)

    def test_build_design_matrix_mismatched_shapes(self, dummy_model):
        """Test that build_design_matrix raises on shape mismatch."""
        x = np.array([100.0, 200.0])
        temp = np.array([160.0, 165.0, 170.0])

        with pytest.raises(ValueError, match="same shape"):
            dummy_model.build_design_matrix(x, temp)

    def test_evaluate(self, dummy_model):
        """Test evaluate method."""
        x = np.array([100.0, 200.0])
        temp = np.array([160.0, 170.0])
        params = np.array([0.5, 10.0])  # y = 0.5*x + 10*dt

        y = dummy_model.evaluate(x, temp, params)

        # dt = (temp - 160) / 10 = [0, 1]
        # y = 0.5*x + 10*dt = [0.5*100 + 0, 0.5*200 + 10] = [50, 110]
        expected = np.array([50.0, 110.0])
        assert np.allclose(y, expected)

    def test_evaluate_scalar_temp(self, dummy_model):
        """Test evaluate with scalar temperature."""
        x = np.array([100.0, 200.0])
        temp = 160.0  # Scalar
        params = np.array([0.5, 10.0])

        y = dummy_model.evaluate(x, temp, params)

        # temp is broadcast to all x
        expected = np.array([50.0, 100.0])
        assert np.allclose(y, expected)


# ============================================================================
# Tests for serialization
# ============================================================================


class TestSerialization:
    """Tests for model serialization and deserialization."""

    def test_as_dict(self, dummy_model):
        """Test as_dict serializes model correctly."""
        header = dummy_model.as_dict()

        assert header["model_name"] == "dummy"
        assert header["model_version"] == "v1"
        assert header["temp_ref"] == 160.0
        assert header["temp_scale"] == 10.0
        assert header["mjd_interval_convention"] == "[start, end)"

    def test_from_header(self):
        """Test from_header reconstructs model correctly."""
        header = {
            "model_name": "dummy",
            "model_version": "v1",
            "temp_ref": 165.0,
            "temp_scale": 5.0,
        }

        model = DummyEquilibriumModel.from_header(header)

        assert model.temp_ref == 165.0
        assert model.temp_scale == 5.0

    def test_from_header_wrong_name(self):
        """Test from_header raises on wrong model name."""
        header = {
            "model_name": "wrong_model",
            "model_version": "v1",
            "temp_ref": 160.0,
            "temp_scale": 10.0,
        }

        with pytest.raises(ValueError, match="Wrong model name"):
            DummyEquilibriumModel.from_header(header)

    def test_from_header_wrong_version(self):
        """Test from_header raises on wrong model version."""
        header = {
            "model_name": "dummy",
            "model_version": "v2",
            "temp_ref": 160.0,
            "temp_scale": 10.0,
        }

        with pytest.raises(ValueError, match="Wrong model version"):
            DummyEquilibriumModel.from_header(header)

    def test_roundtrip_serialization(self, dummy_model):
        """Test that as_dict -> from_header preserves model."""
        header = dummy_model.as_dict()
        reconstructed = DummyEquilibriumModel.from_header(header)

        assert reconstructed.temp_ref == dummy_model.temp_ref
        assert reconstructed.temp_scale == dummy_model.temp_scale

    def test_validate_header_correct(self, dummy_model):
        """Test validate_header with matching header."""
        header = dummy_model.as_dict()
        # Should not raise
        dummy_model.validate_header(header)

    def test_validate_header_wrong_value(self, dummy_model):
        """Test validate_header with mismatched value."""
        header = dummy_model.as_dict()
        header["temp_ref"] = 999.0  # Wrong value

        with pytest.raises(ValueError, match="mismatch"):
            dummy_model.validate_header(header)


# ============================================================================
# Tests for _compare_header_value
# ============================================================================


class TestCompareHeaderValue:
    """Tests for the _compare_header_value helper method."""

    def test_compare_floats_equal(self, dummy_model):
        """Test comparing equal floats."""
        # Should not raise
        dummy_model._compare_header_value("key", 1.0, 1.0)

    def test_compare_floats_close(self, dummy_model):
        """Test comparing floats within tolerance."""
        # Should not raise (within default math.isclose tolerance)
        dummy_model._compare_header_value("key", 1.0, 1.0 + 1e-10)

    def test_compare_floats_different(self, dummy_model):
        """Test comparing different floats raises."""
        with pytest.raises(ValueError, match="mismatch"):
            dummy_model._compare_header_value("key", 1.0, 2.0)

    def test_compare_arrays_equal(self, dummy_model):
        """Test comparing equal arrays."""
        arr = np.array([1.0, 2.0, 3.0])
        # Should not raise
        dummy_model._compare_header_value("key", arr, arr.copy())

    def test_compare_arrays_close(self, dummy_model):
        """Test comparing arrays within tolerance."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = arr1 + 1e-10
        # Should not raise (within np.allclose default tolerance)
        dummy_model._compare_header_value("key", arr1, arr2)

    def test_compare_arrays_different_values(self, dummy_model):
        """Test comparing arrays with different values raises."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.1, 3.0])

        with pytest.raises(ValueError, match="mismatch"):
            dummy_model._compare_header_value("key", arr1, arr2)

    def test_compare_arrays_different_shapes(self, dummy_model):
        """Test comparing arrays with different shapes raises."""
        arr1 = np.array([1.0, 2.0])
        arr2 = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="mismatch"):
            dummy_model._compare_header_value("key", arr1, arr2)

    def test_compare_other_types_equal(self, dummy_model):
        """Test comparing other types (int, str, etc.)."""
        # Should not raise
        dummy_model._compare_header_value("key", 5, 5)
        dummy_model._compare_header_value("key", "test", "test")
        dummy_model._compare_header_value("key", [1, 2], [1, 2])

    def test_compare_other_types_different(self, dummy_model):
        """Test comparing different other types raises."""
        with pytest.raises(ValueError, match="mismatch"):
            dummy_model._compare_header_value("key", 5, 6)

        with pytest.raises(ValueError, match="mismatch"):
            dummy_model._compare_header_value("key", "a", "b")


# ============================================================================
# Tests for tabulation and make_eq_func
# ============================================================================


class TestTabulation:
    """Tests for grid generation and equilibrium function creation."""

    def test_default_tabulation_grid_geomspace(self, dummy_model):
        """Test default grid with positive xmin (geometric spacing)."""
        grid = dummy_model.default_tabulation_grid(n=10, xmin=1.0, xmax=1000.0)

        assert len(grid) == 10
        assert grid[0] == 1.0
        assert grid[-1] == 1000.0
        # Check it's geometric (ratios should be constant)
        ratios = grid[1:] / grid[:-1]
        assert np.allclose(ratios, ratios[0])

    def test_default_tabulation_grid_linspace(self, dummy_model):
        """Test default grid with xmin <= 0 (linear spacing)."""
        grid = dummy_model.default_tabulation_grid(n=10, xmin=-10.0, xmax=10.0)

        assert len(grid) == 10
        assert grid[0] == -10.0
        assert grid[-1] == 10.0
        # Check it's linear (differences should be constant)
        diffs = grid[1:] - grid[:-1]
        assert np.allclose(diffs, diffs[0])

    def test_default_tabulation_grid_invalid_range(self, dummy_model):
        """Test that invalid range raises error."""
        with pytest.raises(ValueError, match="xmin.*xmax"):
            dummy_model.default_tabulation_grid(n=10, xmin=100.0, xmax=10.0)

    def test_make_eq_func(self, dummy_model):
        """Test make_eq_func creates a JaxEqFunc."""
        params = np.array([0.5, 10.0])
        ccd_temp = 160.0

        eq_func = dummy_model.make_eq_func(params, ccd_temp)

        assert isinstance(eq_func, JaxEqFunc)
        assert eq_func.x_grid.shape[0] == 100  # default n=100

    def test_make_eq_func_custom_grid(self, dummy_model):
        """Test make_eq_func with custom tabulation grid."""
        params = np.array([0.5, 10.0])
        ccd_temp = 160.0
        custom_grid = np.linspace(0, 100, 50)

        eq_func = dummy_model.make_eq_func(
            params, ccd_temp, tabulation_grid=custom_grid
        )

        assert eq_func.x_grid.shape[0] == 50
        assert jnp.isclose(eq_func.x_grid[0], 0.0)
        assert jnp.isclose(eq_func.x_grid[-1], 100.0)

    def test_make_eq_func_evaluates_correctly(self, dummy_model):
        """Test that created eq_func evaluates correctly."""
        params = np.array([0.5, 10.0])
        ccd_temp = 160.0

        eq_func = dummy_model.make_eq_func(params, ccd_temp)

        # Evaluate at a point
        x_test = 100.0
        y = eq_func(x_test)

        # Should match evaluate() result
        y_expected = dummy_model.evaluate(
            np.array([x_test]), np.array([ccd_temp]), params
        )[0]

        # Allow some tolerance due to interpolation
        assert jnp.isclose(y, y_expected, atol=0.1)

    def test_make_eq_func_invalid_params(self, dummy_model):
        """Test that make_eq_func raises on invalid params."""
        params = np.array([1.0, 2.0, 3.0])  # Wrong shape
        ccd_temp = 160.0

        with pytest.raises(ValueError, match="expected .* got"):
            dummy_model.make_eq_func(params, ccd_temp)


# ============================================================================
# Integration tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self):
        """Test complete workflow: create, serialize, load, evaluate."""
        # Create model
        model = DummyEquilibriumModel(temp_ref=160.0, temp_scale=10.0)

        # Serialize
        header = model.as_dict()

        # Deserialize
        model2 = DummyEquilibriumModel.from_header(header)

        # Validate
        model2.validate_header(header)

        # Fit params
        params = np.array([0.5, 10.0])

        # Evaluate
        x = np.array([100.0, 200.0, 300.0])
        temp = np.array([160.0, 165.0, 170.0])
        y = model2.evaluate(x, temp, params)

        # Create equilibrium function
        eq_func = model2.make_eq_func(params, ccd_temp=165.0)

        # Evaluate eq_func
        y_func = eq_func(200.0)

        assert isinstance(y, np.ndarray)
        assert jnp.isscalar(y_func)
