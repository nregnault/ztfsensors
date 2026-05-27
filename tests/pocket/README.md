# Tests for Pocket Effect Equilibrium Models

This directory contains comprehensive unit tests for the ZTF sensor pocket effect equilibrium models.

## Test Organization

Tests are organized by module:

- **`test_base.py`**: Tests for `JaxEqFunc` and `BaseEquilibriumModel` abstract classes (43 tests) ✅
- **`test_factory.py`**: Tests for the model factory and registry (26 tests) ✅
- **`test_poly_temp_eq_model.py`**: Tests for polynomial temperature equilibrium model (TODO)
- **`test_spline_temp_eq_model.py`**: Tests for spline temperature equilibrium model (TODO)
- **`test_db.py`**: Tests for equilibrium function database (TODO)
- **`test_fit.py`**: Tests for fitting utilities and diagnostics (TODO)

## Requirements

The tests require the following packages:

```bash
pytest>=7.0
pytest-cov  # for coverage reports
numpy
scipy
jax
equinox
bbf  # for BSpline
polars
pyyaml
saltworks
sksparse
```

All dependencies are managed by pixi.

## Running the Tests

### Run all pocket tests

```bash
# From the project root
pixi run pytest tests/pocket/

# Or from this directory
cd tests/pocket
pixi run pytest
```

### Run a specific test file

```bash
pixi run pytest tests/pocket/test_base.py
pixi run pytest tests/pocket/test_factory.py
```

### Run a specific test class

```bash
pixi run pytest tests/pocket/test_base.py::TestJaxEqFunc
```

### Run a specific test function

```bash
pixi run pytest tests/pocket/test_base.py::TestJaxEqFunc::test_interpolation_at_grid_points
```

### Run with verbose output

```bash
pixi run pytest tests/pocket/ -v
```

### Run with coverage report

```bash
pixi run pytest tests/pocket/ --cov=ztfsensors.pocket --cov-report=html
```

Then open `htmlcov/index.html` in your browser to view the coverage report.

### Run with print statements visible

```bash
pixi run pytest tests/pocket/ -s
```

### Run only tests matching a pattern

```bash
pixi run pytest tests/pocket/ -k "serialization"
```

## Test Structure

### Fixtures (`conftest.py`)

Common fixtures are defined in `conftest.py` and are available to all test modules:

- **Data fixtures**: `sky_levels`, `temperatures`, `synthetic_data`
- **Grid fixtures**: `simple_x_grid`, `jax_grid`
- **Parameter fixtures**: `dummy_params_2d`, `dummy_params_1d`
- **File system fixtures**: `temp_dir`, `temp_file_prefix`
- **Configuration fixtures**: `spline_config`, `poly_config`
- **Database fixtures**: `sample_fit_records`, `mjd_intervals`

### Test Organization

Each test file follows this structure:

```python
# Fixtures and helper classes specific to this module
@pytest.fixture
def my_fixture():
    ...

# Test classes grouped by functionality
class TestFeatureName:
    def test_specific_behavior(self, fixture):
        # Arrange
        ...
        # Act
        ...
        # Assert
        assert ...
```

## Current Test Coverage

**Total: 69 tests passing** ✅

### test_base.py - 43 tests

- **JaxEqFunc (7 tests):**
  - Initialization and attribute storage
  - Interpolation at grid points and between points
  - Clipping behavior (below x_min, above x_max)
  - Vectorized evaluation
  - Support for different input types

- **BaseEquilibriumModel (11 tests):**
  - Required class attributes
  - Parameter validation (shape, type checking)
  - Temperature rescaling with/without attributes
  - Design matrix construction
  - Model evaluation (scalar and vectorial temperature)

- **Serialization (7 tests):**
  - `as_dict()` serialization
  - `from_header()` deserialization
  - Error handling (wrong name/version)
  - Roundtrip consistency
  - Header validation

- **Header Comparison (9 tests):**
  - Float comparison (exact, close, different)
  - Array comparison (shape and value matching)
  - Generic type comparison

- **Tabulation (7 tests):**
  - Grid generation (geometric and linear)
  - Range validation
  - Equilibrium function creation
  - Evaluation correctness

- **Integration (1 test):**
  - Full workflow end-to-end test

### test_factory.py - 26 tests

- **Registration (6 tests):**
  - Single model registration
  - Decorator usage (`@register`)
  - Multiple models and versions
  - Duplicate detection
  - Global registry behavior

- **Instantiation (7 tests):**
  - Basic header parsing
  - Version selection
  - Error handling (unknown model, missing fields)
  - Delegation to model classes

- **Real Models (6 tests):**
  - PolyTempEqModel and SplineTempEqModel registration
  - Model creation from headers
  - Roundtrip serialization

- **Integration (3 tests):**
  - Complete workflows
  - Multiple model coexistence

- **Edge Cases (4 tests):**
  - Empty registry
  - Extra header fields
  - Missing required fields
  - Registry key format validation

## Writing New Tests

### Test naming conventions

- Test files: `test_<module_name>.py`
- Test classes: `Test<FeatureName>`
- Test functions: `test_<specific_behavior>`

### Example test

```python
def test_model_validates_params(dummy_model):
    """Test that model correctly validates parameter shapes."""
    # Arrange
    params = np.array([1.0, 2.0])
    
    # Act
    validated = dummy_model.validate_params(params)
    
    # Assert
    assert validated.shape == (2,)
    assert np.allclose(validated, params)
```

### Testing exceptions

```python
def test_invalid_params_raises(dummy_model):
    """Test that invalid params raise ValueError."""
    params = np.array([1.0, 2.0, 3.0])  # Wrong shape
    
    with pytest.raises(ValueError, match="expected .* got"):
        dummy_model.validate_params(params)
```

### Using parametrize for multiple test cases

```python
@pytest.mark.parametrize("temp_ref,temp_scale,expected", [
    (160.0, 1.0, 0.0),
    (160.0, 10.0, 0.0),
    (150.0, 10.0, 1.0),
])
def test_rescale_temp(temp_ref, temp_scale, expected):
    """Test temperature rescaling with different parameters."""
    model = DummyModel(temp_ref=temp_ref, temp_scale=temp_scale)
    result = model.rescale_temp(160.0)
    assert np.isclose(result, expected)
```

## Test Coverage Goals

We aim for:

- **90%+ code coverage** for all model modules
- **100% coverage** for critical paths (serialization, validation)
- **Edge case coverage**: invalid inputs, boundary conditions, numerical edge cases

## Debugging Failed Tests

### Get detailed error information

```bash
pixi run pytest tests/pocket/ -vv --tb=long
```

### Drop into debugger on failure

```bash
pixi run pytest tests/pocket/ --pdb
```

### Run only failed tests from last run

```bash
pixi run pytest tests/pocket/ --lf
```

### Run failed tests first, then others

```bash
pixi run pytest tests/pocket/ --ff
```

## Best Practices

1. **Keep tests independent**: Each test should be able to run in isolation
2. **Use fixtures**: Reuse common setup code via fixtures
3. **Test one thing**: Each test should focus on a single behavior
4. **Clear assertions**: Use descriptive assertion messages
5. **Test edge cases**: Don't just test the happy path
6. **Mock external dependencies**: Use mocks for file I/O, network calls, etc.
7. **Use meaningful names**: Test names should describe what they test

## Contributing

When adding new features to the pocket effect models:

1. Write tests first (TDD approach recommended)
2. Ensure all tests pass: `pixi run pytest tests/pocket/`
3. Check coverage: `pixi run pytest tests/pocket/ --cov`
4. Update this README if adding new test modules

---

For questions or issues with tests, please open an issue on GitHub.
