
import numpy as np
import pylab as pl
from ztfsensors import sims, PocketModel
from ztfsensors import pocket


def test_1d_pocket_corr(size=250, noise=True):
    """
    """
    pocket_model = PocketModel(1.74, 3056., 0.39, 250000.)
    psf = sims.GaussianPSF1D()
    line = sims.Line(size=size, skylev=200.)
    stars = line.gen_stars(8)
    if noise:
        line.add_stars(stars, psf).add_noise().distort(pocket_model)
    else:
        line.add_stars(stars, psf).distort(pocket_model)
    cs, delta, mask = pocket.correct_1d(pocket_model, line.distorted_data)

    line.plot(reco=delta)

    line.plot()
    # plot the correction

    # line.plot()
    # fig = pl.gcf()
    # fig.axes[0].plot(cs, 'g.:')
    # fig.axes[1].plot(delta, 'g.:')

    
