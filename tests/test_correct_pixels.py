'''
Created on 8 avr. 2024

@author: jcolley
'''

#import numpy as np
#import pylab as pl
#import matplotlib.pyplot as plt
from ztfsensors import sims, PocketModel
from ztfsensors.correct import correct_pixels


def test_correct_pixels(shape=(1000,1000), skylev=322.):
    """
    """
    pocket_model = PocketModel(1.74, 3056., 0.39, 250000.)
    psf = sims.GaussianPSF2D()
    im = sims.Image(shape=shape, skylev=skylev)
    stars = im.gen_stars(250)
    im.add_stars(stars, psf).add_noise().distort(pocket_model)

    cs  = correct_pixels(pocket_model, im.distorted_data)

    if False:
        im.plot()
    
        fig, axes = pl.subplots(nrows=1, ncols=2, figsize=(12, 8),
                                sharex=True, sharey=True)
        axes[0].imshow(cs-im.true_data, vmin=-0.1, vmax=0.1)
        axes[0].set_xlabel('x [pixels]')
        axes[0].set_ylabel('y [pixels]')
        axes[0].set_title('reco - truth')
    # fig.colorbar(im)
    # delta = cs - im.distorted_data
    # axes[1].imshow(delta, vmin=-20, vmax=20.)
    # axes[1].set_xlabel('x [pixels]')
    # axes[1].set_title('correction')
    # fig.suptitle('2D correction')

    #print(cs)

    return cs

if __name__ == '__main__':
    for idx in range(50):
        test_correct_pixels()
    #plt.show()