"""
Tests for pix_distortion.py: invert_line, invert (JAX) and their
numpy equivalents (NumpyEqFunc, invert_numpy).

Structure
---------
Part 1 – JAX backend invariants (pass immediately with the current code):
  - TestInvertLine        : mathematical invariants of the JAX scan-based invert_line
  - TestInvert            : 2-D wrapper (vmap) — invariants and consistency with invert_line

Part 2 – numpy backend (fail until NumpyEqFunc / invert_numpy are implemented):
  - TestNumpyEqFunc       : NumpyEqFunc behaves like JaxEqFunc (clipping, interp, 2-D input)
  - TestInvertNumpy       : invert_numpy behaviour and exact agreement with JAX invert
  - TestCorrectPixels     : correct_pixels dispatches on f_eq type
  - TestMakeEqFuncBackend : BaseEquilibriumModel.make_eq_func(backend=...) kwarg
  - TestDbGetEqFuncBackend: EqFuncDb.get_eq_func(backend=...) kwarg
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ztfsensors.pocket.models.base import JaxEqFunc
from ztfsensors.pocket.pix_distortion import distort_line, invert, invert_line, predict

# ============================================================================
# Shared fixtures
# ============================================================================


@pytest.fixture
def linear_eq_func():
    """f_eq(x) = 0.01 * x  (linear, easy to reason about analytically).

    x_grid starts at 0 so that f_eq(0) = 0 — useful for the zero-input tests.
    """
    alpha = 0.01
    x_grid = jnp.linspace(0.0, 5000.0, 1000, dtype=jnp.float32)
    y_grid = (alpha * x_grid).astype(jnp.float32)
    return JaxEqFunc(x_grid=x_grid, y_grid=y_grid)


@pytest.fixture
def realistic_eq_func():
    """A log-saturating equilibrium function, similar to real ZTF curves.

    x_grid starts at 10 (matches typical sky level lower bound).
    """
    x_grid = jnp.geomspace(10.0, 5000.0, 500, dtype=jnp.float32)
    y_grid = jnp.array(30.0 * jnp.log1p(x_grid / 50.0), dtype=jnp.float32)
    return JaxEqFunc(x_grid=x_grid, y_grid=y_grid)


@pytest.fixture
def flat_row():
    """A 1-D pixel row: constant sky level (200 ADU) + small Gaussian noise."""
    rng = np.random.default_rng(0)
    return (300.0 + rng.normal(0.0, 2.0, 200)).astype(np.float32)


@pytest.fixture
def realistic_image():
    """A 2-D image (30 rows × 200 cols) with sky background and two bright stars."""
    rng = np.random.default_rng(1)
    img = np.full((30, 200), 200.0, dtype=np.float32)
    img += rng.normal(0.0, 3.0, img.shape).astype(np.float32)
    img[5, 40:46] += 800.0  # bright star
    img[15, 120:124] += 400.0  # dimmer star
    return img


# ============================================================================
# Part 1 — JAX backend: mathematical invariants
# ============================================================================


class TestInvertLine:
    """Mathematical invariants of invert_line (JAX lax.scan-based)."""

    def test_output_shape(self, realistic_eq_func, flat_row):
        """Output has the same shape as the input row."""
        d = jnp.array(flat_row)
        result = invert_line(d, realistic_eq_func)
        assert result.shape == d.shape

    def test_zero_row_gives_zero_output(self, linear_eq_func):
        """All-zero input → all-zero output.

        With linear_eq_func: f_eq(0) = 0.
        → delta[0] = f_eq(0) - 0 = 0
        → delta[i] = f_eq(0) - f_eq(0) = 0 for all i
        → output = input + 0 = 0
        """
        d = jnp.zeros(50, dtype=jnp.float32)
        result = invert_line(d, linear_eq_func)
        np.testing.assert_allclose(np.array(result), 0.0, atol=1e-6)

    def test_constant_row_first_pixel_corrected(self, linear_eq_func):
        """For constant input d = v, result[0] = v + f_eq(v).

        The pocket hasn't charged at all before the first pixel, so the
        first delta equals f_eq(v) entirely.
        """
        alpha, v = 0.01, 500.0
        d = jnp.full(50, v, dtype=jnp.float32)
        result = invert_line(d, linear_eq_func)
        expected_first = v + alpha * v  # v + f_eq(v)
        np.testing.assert_allclose(float(result[0]), expected_first, rtol=1e-4)

    def test_constant_row_remaining_pixels_unchanged(self, linear_eq_func):
        """For constant input d = v, result[i > 0] = v.

        When the signal is constant the pocket is at equilibrium after the
        first pixel, so delta[i > 0] = f_eq(v) - f_eq(v) = 0.
        """
        v = 500.0
        d = jnp.full(50, v, dtype=jnp.float32)
        result = invert_line(d, linear_eq_func)
        np.testing.assert_allclose(np.array(result[1:]), v, rtol=1e-4)

    def test_delta_equals_diff_of_feq(self, linear_eq_func):
        """Verify the closed-form: correction[i] = f_eq(d[i]) - f_eq(d[i-1]).

        This is the key identity that allows the scan to be replaced by a
        vectorised numpy diff (used in invert_numpy).

        We use a step-function input so the identity is verifiable with
        scalar f_eq calls (matching exactly what lax.scan does):

            d = [v, v, v, w, w, w]
            corrections = [f_eq(v), 0, 0, f_eq(w)-f_eq(v), 0, 0]
        """
        v, w = 200.0, 800.0
        d = jnp.array([v, v, v, w, w, w], dtype=jnp.float32)

        result = invert_line(d, linear_eq_func)
        corrections = np.array(result - d)

        feq_v = float(linear_eq_func(jnp.array(v, dtype=jnp.float32)))
        feq_w = float(linear_eq_func(jnp.array(w, dtype=jnp.float32)))
        expected = np.array(
            [feq_v, 0.0, 0.0, feq_w - feq_v, 0.0, 0.0], dtype=np.float32
        )
        np.testing.assert_allclose(corrections, expected, atol=1e-5)

    def test_roundtrip_with_distort_line(self, realistic_eq_func):
        """invert_line(distort_line(x, f_eq), f_eq) ≈ x."""
        rng = np.random.default_rng(7)
        true_vals = jnp.array(200.0 + rng.normal(0, 20, 100).astype(np.float32))
        distorted = distort_line(true_vals, realistic_eq_func)
        recovered = invert_line(distorted, realistic_eq_func)
        np.testing.assert_allclose(np.array(recovered), np.array(true_vals), atol=0.5)

    def test_total_correction_equals_last_feq(self, realistic_eq_func, flat_row):
        """Sum of corrections over a row = f_eq(last pixel).

        The corrections are diffs of feq values (telescoping sum):
          delta[0] + ... + delta[n-1]
          = feq(d[0]) + (feq(d[1])-feq(d[0])) + ... + (feq(d[n-1])-feq(d[n-2]))
          = feq(d[n-1])

        Note: individual corrections CAN be negative when a pixel is fainter
        than its predecessor (the pocket discharges, adding charge back).
        """
        d = jnp.array(flat_row)
        result = invert_line(d, realistic_eq_func)
        total_correction = float(jnp.sum(result - d))
        expected = float(realistic_eq_func(d[-1]))
        np.testing.assert_allclose(total_correction, expected, rtol=1e-4)


class TestInvert:
    """Mathematical invariants of invert (JAX vmap over rows, 2-D)."""

    def test_output_shape(self, realistic_eq_func, realistic_image):
        """Output has the same shape as the input image."""
        d = jnp.array(realistic_image)
        result = invert(d, realistic_eq_func)
        assert result.shape == d.shape

    def test_zero_image_gives_zero_output(self, linear_eq_func):
        """All-zero image → all-zero output (f_eq(0) = 0 for linear_eq_func)."""
        d = jnp.zeros((10, 100), dtype=jnp.float32)
        result = invert(d, linear_eq_func)
        np.testing.assert_allclose(np.array(result), 0.0, atol=1e-6)

    def test_each_row_matches_invert_line(self, realistic_eq_func, realistic_image):
        """invert(image)[i] must equal invert_line(image[i]) for every row i."""
        d = jnp.array(realistic_image)
        result_2d = invert(d, realistic_eq_func)
        for i in range(realistic_image.shape[0]):
            result_1d = invert_line(d[i], realistic_eq_func)
            np.testing.assert_allclose(
                np.array(result_2d[i]),
                np.array(result_1d),
                atol=1e-5,
                err_msg=f"Row {i} mismatch between invert and invert_line",
            )

    def test_roundtrip_with_predict(self, realistic_eq_func, realistic_image):
        """invert(predict(image, f_eq), f_eq) ≈ image."""
        true_vals = jnp.array(realistic_image)
        distorted = predict(true_vals, realistic_eq_func)
        recovered = invert(distorted, realistic_eq_func)
        np.testing.assert_allclose(np.array(recovered), np.array(true_vals), atol=0.5)


# ============================================================================
# Part 2 — numpy backend (will fail until NumpyEqFunc / invert_numpy exist)
# ============================================================================


class TestNumpyEqFunc:
    """NumpyEqFunc must behave identically to JaxEqFunc on any input shape."""

    @pytest.fixture
    def numpy_linear_eq_func(self, linear_eq_func):
        from ztfsensors.pocket.models.base import NumpyEqFunc

        return NumpyEqFunc(
            x_grid=np.array(linear_eq_func.x_grid),
            y_grid=np.array(linear_eq_func.y_grid),
        )

    @pytest.fixture
    def numpy_realistic_eq_func(self, realistic_eq_func):
        from ztfsensors.pocket.models.base import NumpyEqFunc

        return NumpyEqFunc(
            x_grid=np.array(realistic_eq_func.x_grid),
            y_grid=np.array(realistic_eq_func.y_grid),
        )

    def test_returns_numpy_array(self, numpy_linear_eq_func):
        """__call__ must return a plain numpy array, not a JAX array."""
        x = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        result = numpy_linear_eq_func(x)
        assert isinstance(result, np.ndarray)

    def test_scalar_matches_jax(self, linear_eq_func, numpy_linear_eq_func):
        """Scalar evaluation must agree with JaxEqFunc."""
        x = 300.0
        np.testing.assert_allclose(
            numpy_linear_eq_func(x), float(linear_eq_func(x)), rtol=1e-5
        )

    def test_1d_array_matches_jax(self, linear_eq_func, numpy_linear_eq_func):
        """1-D array evaluation must agree with JaxEqFunc."""
        x = np.linspace(50.0, 3000.0, 50, dtype=np.float32)
        np.testing.assert_allclose(
            numpy_linear_eq_func(x), np.array(linear_eq_func(x)), rtol=1e-5
        )

    def test_2d_array_matches_jax(self, linear_eq_func, numpy_linear_eq_func):
        """2-D array evaluation must agree with JaxEqFunc.

        This is critical: invert_numpy calls f_eq on the entire 2-D image at
        once, so NumpyEqFunc must handle arbitrary shapes (np.interp does).
        """
        rng = np.random.default_rng(3)
        x = rng.uniform(50.0, 3000.0, (10, 50)).astype(np.float32)
        np.testing.assert_allclose(
            numpy_linear_eq_func(x), np.array(linear_eq_func(x)), rtol=1e-5
        )

    def test_clipping_below_xmin_matches_jax(
        self, linear_eq_func, numpy_linear_eq_func
    ):
        """Values below x_grid[0] must be clipped, matching JaxEqFunc."""
        x = np.array([-500.0], dtype=np.float32)
        np.testing.assert_allclose(
            numpy_linear_eq_func(x), np.array(linear_eq_func(x)), rtol=1e-5
        )

    def test_clipping_above_xmax_matches_jax(
        self, linear_eq_func, numpy_linear_eq_func
    ):
        """Values above x_max (peak of y_grid) must be clipped, matching JaxEqFunc."""
        x = np.array([99999.0], dtype=np.float32)
        np.testing.assert_allclose(
            numpy_linear_eq_func(x), np.array(linear_eq_func(x)), rtol=1e-5
        )

    def test_realistic_1d_matches_jax(self, realistic_eq_func, numpy_realistic_eq_func):
        """Agreement for the realistic (log-saturating) curve on a 1-D array."""
        x = np.geomspace(15.0, 4000.0, 60, dtype=np.float32)
        np.testing.assert_allclose(
            numpy_realistic_eq_func(x), np.array(realistic_eq_func(x)), rtol=1e-5
        )

    def test_realistic_2d_matches_jax(self, realistic_eq_func, numpy_realistic_eq_func):
        """Agreement for the realistic curve on a 2-D array."""
        rng = np.random.default_rng(4)
        x = rng.uniform(15.0, 4000.0, (8, 40)).astype(np.float32)
        np.testing.assert_allclose(
            numpy_realistic_eq_func(x), np.array(realistic_eq_func(x)), rtol=1e-5
        )


class TestInvertNumpy:
    """invert_numpy must reproduce the JAX invert results exactly."""

    @pytest.fixture
    def numpy_realistic_eq_func(self, realistic_eq_func):
        from ztfsensors.pocket.models.base import NumpyEqFunc

        return NumpyEqFunc(
            x_grid=np.array(realistic_eq_func.x_grid),
            y_grid=np.array(realistic_eq_func.y_grid),
        )

    @pytest.fixture
    def numpy_linear_eq_func(self, linear_eq_func):
        from ztfsensors.pocket.models.base import NumpyEqFunc

        return NumpyEqFunc(
            x_grid=np.array(linear_eq_func.x_grid),
            y_grid=np.array(linear_eq_func.y_grid),
        )

    def test_output_shape(self, numpy_realistic_eq_func, realistic_image):
        """Output must have the same shape as the input."""
        from ztfsensors.pocket.pix_distortion import invert_numpy

        result = invert_numpy(realistic_image, numpy_realistic_eq_func)
        assert result.shape == realistic_image.shape

    def test_returns_numpy_array(self, numpy_realistic_eq_func, realistic_image):
        """invert_numpy must return a plain numpy array."""
        from ztfsensors.pocket.pix_distortion import invert_numpy

        result = invert_numpy(realistic_image, numpy_realistic_eq_func)
        assert isinstance(result, np.ndarray)

    def test_zero_image_gives_zero_output(self, numpy_linear_eq_func):
        """All-zero image → all-zero output (f_eq(0) = 0 for linear_eq_func)."""
        from ztfsensors.pocket.pix_distortion import invert_numpy

        d = np.zeros((5, 50), dtype=np.float32)
        result = invert_numpy(d, numpy_linear_eq_func)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_matches_jax_on_flat_image(
        self, numpy_realistic_eq_func, realistic_eq_func, flat_row
    ):
        """invert_numpy and JAX invert agree on a tiled flat-sky image."""
        from ztfsensors.pocket.pix_distortion import invert_numpy

        d = np.tile(flat_row, (5, 1))
        result_np = invert_numpy(d, numpy_realistic_eq_func)
        result_jax = np.array(invert(jnp.array(d), realistic_eq_func))
        np.testing.assert_allclose(result_np, result_jax, rtol=1e-5)

    def test_matches_jax_on_realistic_image(
        self, numpy_realistic_eq_func, realistic_eq_func, realistic_image
    ):
        """invert_numpy and JAX invert agree on a realistic 2-D image."""
        from ztfsensors.pocket.pix_distortion import invert_numpy

        result_np = invert_numpy(realistic_image, numpy_realistic_eq_func)
        result_jax = np.array(invert(jnp.array(realistic_image), realistic_eq_func))
        np.testing.assert_allclose(result_np, result_jax, rtol=1e-5)

    def test_matches_jax_row_by_row(
        self, numpy_realistic_eq_func, realistic_eq_func, realistic_image
    ):
        """Each row of invert_numpy matches the corresponding JAX invert_line."""
        from ztfsensors.pocket.pix_distortion import invert_numpy

        result_np = invert_numpy(realistic_image, numpy_realistic_eq_func)
        for i in range(realistic_image.shape[0]):
            result_jax_row = np.array(
                invert_line(jnp.array(realistic_image[i]), realistic_eq_func)
            )
            np.testing.assert_allclose(
                result_np[i],
                result_jax_row,
                rtol=1e-5,
                err_msg=f"Row {i} mismatch between invert_numpy and JAX invert_line",
            )

    def test_roundtrip_with_jax_distort(
        self, numpy_realistic_eq_func, realistic_eq_func, realistic_image
    ):
        """invert_numpy undoes distortion produced by JAX predict."""
        from ztfsensors.pocket.pix_distortion import invert_numpy

        true_vals = jnp.array(realistic_image)
        distorted_jax = predict(true_vals, realistic_eq_func)
        distorted_np = np.array(distorted_jax)

        recovered = invert_numpy(distorted_np, numpy_realistic_eq_func)
        np.testing.assert_allclose(recovered, realistic_image, atol=0.5)


class TestCorrectPixels:
    """correct_pixels must dispatch to JAX or numpy based on the type of f_eq."""

    @pytest.fixture
    def numpy_realistic_eq_func(self, realistic_eq_func):
        from ztfsensors.pocket.models.base import NumpyEqFunc

        return NumpyEqFunc(
            x_grid=np.array(realistic_eq_func.x_grid),
            y_grid=np.array(realistic_eq_func.y_grid),
        )

    def test_jax_dispatch_returns_result(self, realistic_eq_func, realistic_image):
        """With a JaxEqFunc, correct_pixels returns a result of the right shape."""
        from ztfsensors.pocket import correct_pixels

        result = correct_pixels(jnp.array(realistic_image), realistic_eq_func)
        assert result.shape == realistic_image.shape

    def test_numpy_dispatch_returns_numpy_array(
        self, numpy_realistic_eq_func, realistic_image
    ):
        """With a NumpyEqFunc, correct_pixels must return a plain numpy array."""
        from ztfsensors.pocket import correct_pixels

        result = correct_pixels(realistic_image, numpy_realistic_eq_func)
        assert isinstance(result, np.ndarray)
        assert result.shape == realistic_image.shape

    def test_numpy_and_jax_dispatch_agree(
        self, numpy_realistic_eq_func, realistic_eq_func, realistic_image
    ):
        """Both dispatch paths must produce the same numerical result."""
        from ztfsensors.pocket import correct_pixels

        result_jax = np.array(
            correct_pixels(jnp.array(realistic_image), realistic_eq_func)
        )
        result_np = correct_pixels(realistic_image, numpy_realistic_eq_func)
        np.testing.assert_allclose(result_np, result_jax, rtol=1e-5)


class TestMakeEqFuncBackend:
    """BaseEquilibriumModel.make_eq_func must accept a backend= kwarg."""

    @pytest.fixture
    def poly_model(self):
        from ztfsensors.pocket.models.poly_temp_eq_model import PolyTempEqModel

        return PolyTempEqModel(
            p_deg=[2, 2],
            q_deg=[2],
            x_knot=200.0,
            temp_ref=160.0,
            temp_scale=5.0,
        )

    @pytest.fixture
    def poly_params(self, poly_model):
        rng = np.random.default_rng(99)
        return rng.normal(0, 0.1, poly_model.params_shape)

    def test_jax_backend_returns_jax_eq_func(self, poly_model, poly_params):
        """make_eq_func(backend='jax') must return a JaxEqFunc."""
        eq_func = poly_model.make_eq_func(poly_params, ccd_temp=160.0, backend="jax")
        assert isinstance(eq_func, JaxEqFunc)

    def test_numpy_backend_returns_numpy_eq_func(self, poly_model, poly_params):
        """make_eq_func(backend='numpy') must return a NumpyEqFunc."""
        from ztfsensors.pocket.models.base import NumpyEqFunc

        eq_func = poly_model.make_eq_func(poly_params, ccd_temp=160.0, backend="numpy")
        assert isinstance(eq_func, NumpyEqFunc)

    def test_numpy_and_jax_backends_agree(self, poly_model, poly_params):
        """JaxEqFunc and NumpyEqFunc from make_eq_func must evaluate identically."""
        grid = np.geomspace(50.0, 5000.0, 50)
        eq_jax = poly_model.make_eq_func(
            poly_params, ccd_temp=160.0, backend="jax", tabulation_grid=grid
        )
        eq_np = poly_model.make_eq_func(
            poly_params, ccd_temp=160.0, backend="numpy", tabulation_grid=grid
        )
        x = np.linspace(60.0, 4000.0, 40, dtype=np.float32)
        np.testing.assert_allclose(eq_np(x), np.array(eq_jax(x)), rtol=1e-5)

    def test_invalid_backend_raises_value_error(self, poly_model, poly_params):
        """Unknown backend name must raise ValueError."""
        with pytest.raises(ValueError, match="backend"):
            poly_model.make_eq_func(poly_params, ccd_temp=160.0, backend="torch")


class TestDbGetEqFuncBackend:
    """EqFuncDb.get_eq_func must accept and forward a backend= kwarg."""

    @pytest.fixture
    def poly_db(self):
        """A minimal in-memory EqFuncDb backed by a PolyTempEqModel."""
        import polars as pl

        from ztfsensors.pocket.db import EqFuncDb
        from ztfsensors.pocket.models.poly_temp_eq_model import PolyTempEqModel

        model = PolyTempEqModel(
            p_deg=[2, 2],
            q_deg=[2],
            x_knot=200.0,
            temp_ref=160.0,
            temp_scale=5.0,
        )
        rng = np.random.default_rng(77)
        params = rng.normal(0, 0.1, model.params_shape).tolist()
        df = pl.DataFrame(
            {
                "ccdid": [1],
                "qid": [1],
                "mjd_start": [58000.0],
                "mjd_end": [59000.0],
                "temp_min": [155.0],
                "temp_max": [165.0],
                "params": [params],
            }
        )
        return EqFuncDb(df=df, model=model)

    @pytest.fixture
    def tabulation_grid(self):
        return np.geomspace(50.0, 5000.0, 50)

    def test_jax_backend_returns_jax_eq_func(self, poly_db, tabulation_grid):
        """get_eq_func(backend='jax') must return a JaxEqFunc."""
        eq_func = poly_db.get_eq_func(
            ccdid=1,
            qid=1,
            mjd=58050.0,
            ccd_temp=160.0,
            tabulation_grid=tabulation_grid,
            backend="jax",
        )
        assert isinstance(eq_func, JaxEqFunc)

    def test_numpy_backend_returns_numpy_eq_func(self, poly_db, tabulation_grid):
        """get_eq_func(backend='numpy') must return a NumpyEqFunc."""
        from ztfsensors.pocket.models.base import NumpyEqFunc

        eq_func = poly_db.get_eq_func(
            ccdid=1,
            qid=1,
            mjd=58050.0,
            ccd_temp=160.0,
            tabulation_grid=tabulation_grid,
            backend="numpy",
        )
        assert isinstance(eq_func, NumpyEqFunc)

    def test_numpy_and_jax_backends_agree(self, poly_db, tabulation_grid):
        """Both backends must evaluate identically on the same tabulation grid."""
        eq_jax = poly_db.get_eq_func(
            ccdid=1,
            qid=1,
            mjd=58050.0,
            ccd_temp=160.0,
            tabulation_grid=tabulation_grid,
            backend="jax",
        )
        eq_np = poly_db.get_eq_func(
            ccdid=1,
            qid=1,
            mjd=58050.0,
            ccd_temp=160.0,
            tabulation_grid=tabulation_grid,
            backend="numpy",
        )
        x = np.linspace(60.0, 4000.0, 40, dtype=np.float32)
        np.testing.assert_allclose(eq_np(x), np.array(eq_jax(x)), rtol=1e-5)

    def test_invalid_backend_raises(self, poly_db, tabulation_grid):
        """Unknown backend must propagate a ValueError from make_eq_func."""
        with pytest.raises(ValueError, match="backend"):
            poly_db.get_eq_func(
                ccdid=1,
                qid=1,
                mjd=58050.0,
                ccd_temp=160.0,
                tabulation_grid=tabulation_grid,
                backend="cupy",
            )
