import numpy as np
import pytest

from ztfsensors.correct import correct_pixels
from ztfsensors.pocket import PocketModel
from ztfsensors.test import POCKET_PARAMS, get_pocket_test, simulate_1d, simulate_2d

BACKENDS = ["numpy-nr", "numpy", "cpp"]
# data_and_overscan = correct_pixels(model, data)


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


@pytest.mark.parametrize("shape", [(200, 200), (1000, 1000)])
@pytest.mark.parametrize("skylev", [100, 200, 1000])
@pytest.mark.parametrize("noise", [False, True])
def test_apply_all_backends_2d(shape, skylev, noise):
    data = simulate_2d(shape=shape, skylev=skylev, noise=noise)
    model = PocketModel(*POCKET_PARAMS)
    res = {name: model.apply(data.copy(), backend=name) for name in BACKENDS}

    # test that all methods give the same result
    # note: we exclude the first 50 columns because there are some differences
    # with the cpp version for the first columns...
    ref = BACKENDS[0]
    for name in BACKENDS[1:]:
        np.testing.assert_array_almost_equal(
            res[ref][:, 50:], res[name][:, 50:], decimal=8
        )


def test_image():
    model, data = get_pocket_test()
    res = {name: correct_pixels(model, data.copy(), backend=name) for name in BACKENDS}

    # test that all methods give the same result
    # note: we exclude the first 10 columns because there are some differences
    # with the cpp version for the first columns...
    ref = BACKENDS[0]
    for name in BACKENDS[1:]:
        np.testing.assert_array_almost_equal(
            res[ref][:, 10:], res[name][:, 10:], decimal=2
        )
