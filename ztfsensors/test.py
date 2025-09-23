import numpy as np
import ztfimg

from ztfsensors import pocket
from ztfsensors.sims import GaussianPSF1D, GaussianPSF2D, Image, Line
from ztfsensors.pocket import PocketModel

POCKET_PARAMS = (1.74, 3056.0, 0.39, 250000.0)


def get_pocket_test(filename="ztf_20200401152477_000517_zg_c06_o.fits.fz", data=False):
    """get the pocket model and a raw-data_and_overscan test case.

    Returns
    -------
    PocketModel, 2d-array
        - pockelmodel
        - pixel

    Example
    -------
    pocket_model, pixels = get_pocket_test()
    %time _ = pocket_model.apply(pixels, backend="numpy")

    """

    # Access the raw quadrant
    rawimg = ztfimg.RawCCD.from_filename(filename, as_path=False)
    quad = rawimg.get_quadrant(1)

    # Get data with overscan at the end
    data_and_overscan = quad.get_data_and_overscan()

    # the model with quadrant's pocket parameter
    pocket_model = pocket.PocketModel(
        **pocket.get_config(quad.ccdid, quad.qid).values[0]
    )
    if data:
        return pocket_model, data_and_overscan

    current_state = data_and_overscan.copy()
    current_state[:, -30:] = 0.0  # set overscan to zero
    current_state[0:2] = np.median(data_and_overscan)  # default

    return pocket_model, current_state


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


def simulate_2d(n_stars=250, shape=(1000, 1000), skylev=200.0, noise=False):
    np.random.seed(42)
    pocket_model = PocketModel(*POCKET_PARAMS)
    psf = GaussianPSF2D()
    im = Image(shape=shape, skylev=skylev)
    im.add_stars(im.gen_stars(n_stars), psf)
    if noise:
        im.add_noise()
    im.distort(pocket_model)
    return im.distorted_data
