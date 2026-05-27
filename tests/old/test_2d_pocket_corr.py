
import pylab as pl
from ztfsensors import sims, PocketModel
from ztfsensors import pocket


def test_2d_pocket_corr(shape=(1000,1000), skylev=322.):
    """
    """
    pocket_model = PocketModel(1.74, 3056., 0.39, 250000.)
    psf = sims.GaussianPSF2D()
    im = sims.Image(shape=shape, skylev=skylev)
    stars = im.gen_stars(250)
    im.add_stars(stars, psf).add_noise().distort(pocket_model)

    cs, delta, mask = pocket.correct_2d(pocket_model, im.distorted_data)

    im.plot()

    fig, axes = pl.subplots(nrows=1, ncols=2, figsize=(12, 8),
                            sharex=True, sharey=True)
    im = axes[0].imshow(cs-im.true_data, vmin=-0.1, vmax=0.1)
    axes[0].set_xlabel('x [pixels]')
    axes[0].set_ylabel('y [pixels]')
    axes[0].set_title('reco - truth')
    # fig.colorbar(im)
    axes[1].imshow(delta, vmin=-20, vmax=20.)
    axes[1].set_xlabel('x [pixels]')
    axes[1].set_title('correction')
    fig.suptitle('2D correction')

    return cs, delta
