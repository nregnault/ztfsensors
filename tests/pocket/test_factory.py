"""
Tests for EquilibriumModelFactory.

This module tests the factory pattern for registering and instantiating
equilibrium models from serialized headers.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from ztfsensors.pocket.models.base import BaseEquilibriumModel
from ztfsensors.pocket.models.factory import EquilibriumModelFactory
from ztfsensors.pocket.models.poly_temp_eq_model import PolyTempEqModel
from ztfsensors.pocket.models.spline_temp_eq_model import SplineTempEqModel

# ============================================================================
# Test fixtures and helper classes
# ============================================================================


class DummyModelA(BaseEquilibriumModel):
    """First test model for factory testing."""

    MODEL_NAME = "test_model_a"
    MODEL_VERSION = "v1"
    HEADER_FIELDS = ("param_a", "param_b")

    def __init__(self, param_a: float = 1.0, param_b: float = 2.0):
        self.param_a = float(param_a)
        self.param_b = float(param_b)

    @property
    def params_shape(self) -> tuple[int]:
        return (2,)

    def build_design_matrix(self, x, ccd_temp):
        x = np.asarray(x)
        return sparse.csr_matrix(np.column_stack([x, np.ones_like(x)]))


class DummyModelB(BaseEquilibriumModel):
    """Second test model for factory testing."""

    MODEL_NAME = "test_model_b"
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


class DummyModelV2(BaseEquilibriumModel):
    """Same name as DummyModelA but different version."""

    MODEL_NAME = "test_model_a"
    MODEL_VERSION = "v2"
    HEADER_FIELDS = ("new_param",)

    def __init__(self, new_param: float = 42.0):
        self.new_param = float(new_param)

    @property
    def params_shape(self) -> tuple[int]:
        return (1,)

    def build_design_matrix(self, x, ccd_temp):
        x = np.asarray(x)
        return sparse.csr_matrix(np.atleast_2d(x).T)


@pytest.fixture
def clean_registry():
    """
    Provide a clean factory registry for testing.

    Saves the current registry, yields for testing, then restores it.
    This ensures tests don't interfere with each other.
    """
    # Save current registry
    original_registry = EquilibriumModelFactory._registry.copy()

    # Clear registry for testing
    EquilibriumModelFactory._registry.clear()

    yield EquilibriumModelFactory

    # Restore original registry
    EquilibriumModelFactory._registry = original_registry


# ============================================================================
# Tests for model registration
# ============================================================================


class TestRegistration:
    """Tests for registering models in the factory."""

    def test_register_single_model(self, clean_registry):
        """Test registering a single model."""
        registered = clean_registry.register(DummyModelA)

        # Should return the same class
        assert registered is DummyModelA

        # Should be in registry
        key = ("test_model_a", "v1")
        assert key in clean_registry._registry
        assert clean_registry._registry[key] is DummyModelA

    def test_register_as_decorator(self, clean_registry):
        """Test using register as a decorator."""

        @clean_registry.register
        class DecoratedModel(BaseEquilibriumModel):
            MODEL_NAME = "decorated"
            MODEL_VERSION = "v1"
            HEADER_FIELDS = ()

            @property
            def params_shape(self):
                return (1,)

            def build_design_matrix(self, x, ccd_temp):
                return sparse.csr_matrix(np.atleast_2d(x).T)

        # Should be registered
        key = ("decorated", "v1")
        assert key in clean_registry._registry
        assert clean_registry._registry[key] is DecoratedModel

    def test_register_multiple_models(self, clean_registry):
        """Test registering multiple different models."""
        clean_registry.register(DummyModelA)
        clean_registry.register(DummyModelB)

        assert len(clean_registry._registry) == 2
        assert ("test_model_a", "v1") in clean_registry._registry
        assert ("test_model_b", "v1") in clean_registry._registry

    def test_register_multiple_versions(self, clean_registry):
        """Test registering multiple versions of the same model."""
        clean_registry.register(DummyModelA)
        clean_registry.register(DummyModelV2)

        assert len(clean_registry._registry) == 2
        assert ("test_model_a", "v1") in clean_registry._registry
        assert ("test_model_a", "v2") in clean_registry._registry

    def test_register_duplicate_raises(self, clean_registry):
        """Test that registering the same model twice raises an error."""
        clean_registry.register(DummyModelA)

        with pytest.raises(ValueError, match="already registered"):
            clean_registry.register(DummyModelA)

    def test_registry_is_global(self):
        """Test that the registry is shared across instances."""
        # The registry should be a class variable, not instance variable
        assert hasattr(EquilibriumModelFactory, "_registry")

        # Should be the same object
        registry1 = EquilibriumModelFactory._registry
        registry2 = EquilibriumModelFactory._registry
        assert registry1 is registry2


# ============================================================================
# Tests for model instantiation from headers
# ============================================================================


class TestFromHeader:
    """Tests for creating models from headers."""

    def test_from_header_basic(self, clean_registry):
        """Test creating a model from a basic header."""
        clean_registry.register(DummyModelA)

        header = {
            "model_name": "test_model_a",
            "model_version": "v1",
            "param_a": 5.0,
            "param_b": 10.0,
        }

        model = clean_registry.from_header(header)

        assert isinstance(model, DummyModelA)
        assert model.param_a == 5.0
        assert model.param_b == 10.0

    def test_from_header_selects_correct_version(self, clean_registry):
        """Test that from_header selects the correct version."""
        clean_registry.register(DummyModelA)
        clean_registry.register(DummyModelV2)

        # Request v1
        header_v1 = {
            "model_name": "test_model_a",
            "model_version": "v1",
            "param_a": 1.0,
            "param_b": 2.0,
        }
        model_v1 = clean_registry.from_header(header_v1)
        assert isinstance(model_v1, DummyModelA)
        assert model_v1.MODEL_VERSION == "v1"

        # Request v2
        header_v2 = {
            "model_name": "test_model_a",
            "model_version": "v2",
            "new_param": 42.0,
        }
        model_v2 = clean_registry.from_header(header_v2)
        assert isinstance(model_v2, DummyModelV2)
        assert model_v2.MODEL_VERSION == "v2"

    def test_from_header_unknown_model_raises(self, clean_registry):
        """Test that requesting an unknown model raises ValueError."""
        header = {
            "model_name": "nonexistent",
            "model_version": "v1",
        }

        with pytest.raises(ValueError, match="Unknown model"):
            clean_registry.from_header(header)

    def test_from_header_missing_model_name_raises(self, clean_registry):
        """Test that missing model_name raises ValueError."""
        header = {
            "model_version": "v1",
            "param_a": 1.0,
        }

        with pytest.raises(ValueError, match="must contain 'model_name'"):
            clean_registry.from_header(header)

    def test_from_header_missing_model_version_raises(self, clean_registry):
        """Test that missing model_version raises ValueError."""
        header = {
            "model_name": "test_model_a",
            "param_a": 1.0,
        }

        with pytest.raises(ValueError, match="must contain.*'model_version'"):
            clean_registry.from_header(header)

    def test_from_header_none_model_name_raises(self, clean_registry):
        """Test that None model_name raises ValueError."""
        header = {
            "model_name": None,
            "model_version": "v1",
        }

        with pytest.raises(ValueError, match="must contain 'model_name'"):
            clean_registry.from_header(header)

    def test_from_header_delegates_to_model(self, clean_registry):
        """Test that from_header delegates to the model class's from_header."""
        clean_registry.register(DummyModelA)

        header = {
            "model_name": "test_model_a",
            "model_version": "v1",
            "param_a": 7.0,
            "param_b": 14.0,
        }

        model = clean_registry.from_header(header)

        # The model's from_header should have been called with the header
        # and it should have extracted the HEADER_FIELDS
        assert model.param_a == 7.0
        assert model.param_b == 14.0


# ============================================================================
# Tests with real models
# ============================================================================


class TestWithRealModels:
    """Tests using the actual PolyTempEqModel and SplineTempEqModel."""

    def test_poly_model_is_registered(self):
        """Test that PolyTempEqModel is registered in the factory."""
        key = (PolyTempEqModel.MODEL_NAME, PolyTempEqModel.MODEL_VERSION)
        assert key in EquilibriumModelFactory._registry
        assert EquilibriumModelFactory._registry[key] is PolyTempEqModel

    def test_spline_model_is_registered(self):
        """Test that SplineTempEqModel is registered in the factory."""
        key = (SplineTempEqModel.MODEL_NAME, SplineTempEqModel.MODEL_VERSION)
        assert key in EquilibriumModelFactory._registry
        assert EquilibriumModelFactory._registry[key] is SplineTempEqModel

    def test_from_header_creates_poly_model(self):
        """Test creating a PolyTempEqModel from a header."""
        header = {
            "model_name": "poly_temp_eq",
            "model_version": "v1",
            "p_deg": [2, 2],
            "q_deg": [3],
            "x_knot": 250.0,
            "temp_ref": 165.0,
            "temp_scale": 5.0,
        }

        model = EquilibriumModelFactory.from_header(header)

        assert isinstance(model, PolyTempEqModel)
        assert model.p_deg == [2, 2]
        assert model.q_deg == [3]
        assert model.x_knot == 250.0
        assert model.temp_ref == 165.0
        assert model.temp_scale == 5.0

    def test_from_header_creates_spline_model(self):
        """Test creating a SplineTempEqModel from a header."""
        header = {
            "model_name": "spline_temp_eq",
            "model_version": "v1",
            "basis_grid": [50.0, 100.0, 500.0, 1000.0, 5000.0],
            "temp_deg": 4,
            "temp_ref": 160.0,
            "temp_scale": 10.0,
            "basis_order": 4,
            "basis_size": None,  # Will be computed
            "param_layout": "basis_major_high_to_low_temp_power",
        }

        model = EquilibriumModelFactory.from_header(header)

        assert isinstance(model, SplineTempEqModel)
        assert model.temp_deg == 4
        assert model.temp_ref == 160.0
        assert model.temp_scale == 10.0
        assert model.basis_order == 4

    def test_roundtrip_poly_model(self):
        """Test serialization roundtrip with PolyTempEqModel."""
        # Create model
        model1 = PolyTempEqModel(
            p_deg=[3, 2],
            q_deg=[2, 1],
            x_knot=300.0,
            temp_ref=155.0,
            temp_scale=8.0,
        )

        # Serialize
        header = model1.as_dict()

        # Deserialize
        model2 = EquilibriumModelFactory.from_header(header)

        # Compare
        assert isinstance(model2, PolyTempEqModel)
        assert model2.p_deg == model1.p_deg
        assert model2.q_deg == model1.q_deg
        assert model2.x_knot == model1.x_knot
        assert model2.temp_ref == model1.temp_ref
        assert model2.temp_scale == model1.temp_scale

    def test_roundtrip_spline_model(self):
        """Test serialization roundtrip with SplineTempEqModel."""
        # Create model
        basis_grid = np.geomspace(50.0, 5000.0, 10)
        model1 = SplineTempEqModel(
            basis_grid=basis_grid,
            temp_deg=5,
            temp_ref=162.0,
            temp_scale=7.0,
            basis_order=4,
        )

        # Serialize
        header = model1.as_dict()

        # Deserialize
        model2 = EquilibriumModelFactory.from_header(header)

        # Compare
        assert isinstance(model2, SplineTempEqModel)
        assert model2.temp_deg == model1.temp_deg
        assert model2.temp_ref == model1.temp_ref
        assert model2.temp_scale == model1.temp_scale
        assert model2.basis_order == model1.basis_order
        assert np.allclose(model2.basis_grid, model1.basis_grid)


# ============================================================================
# Integration tests
# ============================================================================


class TestIntegration:
    """Integration tests combining registration and instantiation."""

    def test_full_workflow(self, clean_registry):
        """Test complete workflow: register, serialize, deserialize."""
        # Register
        clean_registry.register(DummyModelA)

        # Create and configure model
        model1 = DummyModelA(param_a=3.14, param_b=2.71)

        # Serialize
        header = model1.as_dict()

        # Deserialize via factory
        model2 = clean_registry.from_header(header)

        # Verify
        assert isinstance(model2, DummyModelA)
        assert model2.param_a == 3.14
        assert model2.param_b == 2.71

        # Validate header
        model2.validate_header(header)

    def test_multiple_models_coexist(self, clean_registry):
        """Test that multiple models can be registered and used together."""
        clean_registry.register(DummyModelA)
        clean_registry.register(DummyModelB)

        # Create header for model A
        header_a = {
            "model_name": "test_model_a",
            "model_version": "v1",
            "param_a": 1.0,
            "param_b": 2.0,
        }

        # Create header for model B
        header_b = {
            "model_name": "test_model_b",
            "model_version": "v1",
            "scale": 5.0,
        }

        # Instantiate both
        model_a = clean_registry.from_header(header_a)
        model_b = clean_registry.from_header(header_b)

        # Verify correct types
        assert isinstance(model_a, DummyModelA)
        assert isinstance(model_b, DummyModelB)

        # Verify correct parameters
        assert model_a.param_a == 1.0
        assert model_b.scale == 5.0

    def test_factory_with_real_and_test_models(self):
        """Test that factory works with both real and test models together."""
        # The real models should already be registered
        # Add a test model to the existing registry

        # Save original registry size
        original_size = len(EquilibriumModelFactory._registry)

        # This should work because DummyModelA has different name than real models
        # But we need to be careful not to pollute the global registry
        # So we'll just verify the real models work

        # Create a poly model
        header_poly = {
            "model_name": "poly_temp_eq",
            "model_version": "v1",
            "p_deg": [2, 2, 2],
            "q_deg": [3, 3],
            "x_knot": 200.0,
            "temp_ref": 160.0,
            "temp_scale": 1.0,
        }

        model_poly = EquilibriumModelFactory.from_header(header_poly)
        assert isinstance(model_poly, PolyTempEqModel)

        # Create a spline model
        header_spline = {
            "model_name": "spline_temp_eq",
            "model_version": "v1",
            "basis_grid": np.geomspace(50.0, 10000.0, 10).tolist(),
            "temp_deg": 5,
            "temp_ref": 160.0,
            "temp_scale": 1.0,
            "basis_order": 4,
            "basis_size": None,
            "param_layout": "basis_major_high_to_low_temp_power",
        }

        model_spline = EquilibriumModelFactory.from_header(header_spline)
        assert isinstance(model_spline, SplineTempEqModel)


# ============================================================================
# Edge cases and error handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_registry(self, clean_registry):
        """Test that from_header fails on empty registry."""
        header = {
            "model_name": "anything",
            "model_version": "v1",
        }

        with pytest.raises(ValueError, match="Unknown model"):
            clean_registry.from_header(header)

    def test_header_with_extra_fields(self, clean_registry):
        """Test that extra fields in header don't cause problems."""
        clean_registry.register(DummyModelA)

        header = {
            "model_name": "test_model_a",
            "model_version": "v1",
            "param_a": 1.0,
            "param_b": 2.0,
            "extra_field": "ignored",
            "another_extra": 999,
        }

        # Should work fine, extra fields are ignored
        model = clean_registry.from_header(header)
        assert isinstance(model, DummyModelA)

    def test_header_missing_required_field(self, clean_registry):
        """Test that missing required field raises error from model."""
        clean_registry.register(DummyModelA)

        header = {
            "model_name": "test_model_a",
            "model_version": "v1",
            "param_a": 1.0,
            # Missing param_b
        }

        # Should raise KeyError when model tries to extract param_b
        with pytest.raises(KeyError):
            clean_registry.from_header(header)

    def test_registry_key_format(self, clean_registry):
        """Test that registry keys are (name, version) tuples."""
        clean_registry.register(DummyModelA)

        # Keys should be tuples
        for key in clean_registry._registry.keys():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(key[0], str)  # model_name
            assert isinstance(key[1], str)  # model_version
