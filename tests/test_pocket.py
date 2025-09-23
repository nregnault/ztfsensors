import numpy as np
import pytest

from ztfsensors.pocket import PocketModel
from ztfsensors.sims import GaussianPSF1D, Line

BACKENDS = ["numpy-nr", "numpy", "cpp"]
POCKET_PARAMS = (1.74, 3056.0, 0.39, 250000.0)
# data_and_overscan = correct_pixels(model, data)


def simulate_1d(n_stars=5, size=250, skylev=200.0, noise=False):
    np.random.seed(42)
    pocket_model = PocketModel(*POCKET_PARAMS)
    psf = GaussianPSF1D()
    line = Line(size=size, skylev=skylev)
    line.add_stars(line.gen_stars(n_stars), psf)
    if noise:
        line.add_noise()
    line.distort(pocket_model)
    return line.distorted_data


@pytest.mark.parametrize("size", [100, 350, 1000])
@pytest.mark.parametrize("skylev", [100, 200, 1000])
@pytest.mark.parametrize("noise", [False, True])
def test_apply_all_backends_1d(size, skylev, noise):
    data = simulate_1d(n_stars=5, size=size, skylev=skylev, noise=noise)
    model = PocketModel(*POCKET_PARAMS)
    res = {name: model.apply(data.copy(), backend=name) for name in BACKENDS}

    # test that all methods give the same result
    ref = BACKENDS[0]
    for name in BACKENDS[1:]:
        np.testing.assert_array_almost_equal(res[ref], res[name], decimal=11)
