"""Simulated data
"""

import numpy as np
import pylab as pl


class GaussianPSF1D:

    def __init__(self, sigma=3.):
        self.sigma = sigma
        self.nx = int(2. * 5. * sigma)
        if self.nx % 2 == 1:
            self.nx += 1
        self.delta = int(self.nx/2)
        self.grid = np.linspace(-self.delta+0.5, self.delta-0.5, self.nx)
        self.norm = 1. / (np.sqrt(2. * np.pi) * self.sigma)

    def __call__(self, p, nx=None):
        """
        """
        x, flux = p
        ix = int(np.fix(x))
        dx = x - ix

        vals = flux * self.norm * np.exp(-0.5 * (dx+self.grid)**2 / self.sigma**2)
        if nx is None:
            return vals

        xmin = int(max(ix-self.delta, 0))
        xmax = int(min(ix+self.delta, nx))
        im_slice = slice(xmin, xmax, 1)
        psf_slice = slice(int(xmin-ix+self.delta), int(xmax-ix+self.delta), 1)

        return vals[psf_slice], im_slice


class Line:
    """Simplistic simulation of an image line
    """
    def __init__(self, size=1000, skylev=100, overscan_width=30, stars=None, noise=False):
        self.size = size
        self.skylev = skylev
        self.overscan_width = overscan_width
        self.stars = stars

        # self.orig_data = np.zeros(size + overscan_width)
        # self.orig_data[:size] += skylev
        # self.true_data = None
        # self.distorted_data = None
        self.reset()
        if stars:
            psf = GaussianPSF1D()
            self.add_stars(self.stars, psf)
        if noise:
            self.add_noise()

    def reset(self):
        self.orig_data = np.zeros(self.size + self.overscan_width)
        self.orig_data[:self.size] += self.skylev
        self.true_data = None
        self.distorted_data = None

    def add_stars(self, stars, psf):
        self.reset()
        for s in stars:
            vals, slc = psf(s, self.size)
            self.orig_data[slc] += vals
        return self

    def add_noise(self):
        self.true_data = np.zeros_like(self.orig_data)
        self.true_data[:self.size] = self.orig_data[:self.size] + \
            np.random.normal(loc=0., scale=np.sqrt(self.orig_data[:self.size]), size=self.size)
        self.distored_data = None
        return self

    def gen_stars(self, n):
        x = np.random.uniform(0, self.size, n)
        flux = np.random.uniform(0, 100000., n)
        l = np.rec.fromarrays((x, flux), names=['x', 'flux'])
        return l

    def distort(self, model):
        """
        """
        if not hasattr(self, 'true_data') or self.true_data is None:
            self.true_data = self.orig_data.copy()
        self.distorted_data = model.apply(self.true_data)
        return self

    def plot(self, reco=None):
        if reco is not None:
            fig, axes = pl.subplots(nrows=3, ncols=1, sharex=True, figsize=(12,12))
        else:
            fig, axes = pl.subplots(nrows=2, ncols=1, sharex=True, figsize=(12,12))

        axes[0].plot(self.orig_data, 'k:', label='orig data')
        if self.true_data is not None:
            axes[0].plot(self.true_data, 'k.', label='orig + noise (truth)')
        if self.distorted_data is not None:
            axes[0].plot(self.distorted_data, 'r-.', label='distorted')
        if reco is not None:
            axes[0].plot(reco + self.distorted_data, 'b.', label='reconstructed')
        axes[0].set_ylabel('flux')
        axes[0].legend(loc='best')

        if self.distorted_data is not None:
            axes[1].plot(self.distorted_data - self.true_data, 'r.', label='distortion')
        if reco is not None:
            axes[1].plot(reco, 'b.', label='correction')
        axes[1].legend(loc='best')
        axes[1].set_ylabel('distorted - true')
        axes[1].set_xlabel('x')

        if reco is not None:
            axes[2].plot(reco + self.distorted_data - self.true_data, 'b.')
            axes[2].set_xlabel('x')
            axes[2].set_ylabel('sum')
            axes[2].set_ylim((-100,100))

        pl.subplots_adjust(hspace=0.005)

class GaussianPSF2D:

    def __init__(self, sigma=3.):
        """
        """
        self.sigma = sigma
        self.nx = int(2. * 5. * sigma)
        if self.nx % 2 == 1:
            self.nx += 1
        self.delta = int(self.nx/2)
        self.grid_x, self.grid_y = np.mgrid[-self.delta:self.delta,
                                            -self.delta:self.delta]
        self.norm = 1. / (2 * np.pi * self.sigma**2)

    def __call__(self, p, shape=None):
        x, y, flux = p
        ix = int(np.fix(x))
        dx = x - ix
        iy = int(np.fix(y))
        dy = y - iy

        r2 = (dx+self.grid_x)**2 + (dy+self.grid_y)**2
        vals = flux * self.norm * np.exp(-0.5 * r2/self.sigma**2)
        if shape is None:
            return vals

        nx, ny = shape
        xmin = int(max(ix-self.delta, 0))
        xmax = int(min(ix+self.delta, nx))
        im_x_slice = slice(xmin, xmax, 1)
        psf_x_slice = slice(int(xmin-ix+self.delta), int(xmax-ix+self.delta), 1)

        ymin = int(max(iy-self.delta, 0))
        ymax = int(min(iy+self.delta, ny))
        im_y_slice = slice(ymin, ymax, 1)
        psf_y_slice = slice(int(ymin-iy+self.delta), int(ymax-iy+self.delta), 1)

        return vals[psf_x_slice, psf_y_slice], im_x_slice, im_y_slice


class Image:
    """Simplistic 2D image, with an overscan and stars
    """
    def __init__(self, shape=(1000,1000), skylev=100, stars=None, overscan_width=30):
        self.shape = shape
        self.skylev = skylev
        self.stars = stars
        self.overscan_width = overscan_width
        nrows, ncols = self.shape
        self.orig_data = np.zeros((nrows, ncols+self.overscan_width))
        self.orig_data[:,:ncols] = skylev
        self.true_data = None
        self.distorted_data = None

    def add_stars(self, stars, psf):
        for s in stars:
            vals, x_slc, y_slc = psf(s, self.shape)
            self.orig_data[x_slc, y_slc] += vals
        return self

    def add_noise(self):
        self.distorted_data = None
        self.true_data = self.orig_data + \
            np.random.normal(loc=0., scale=np.sqrt(self.orig_data),
                             size=self.orig_data.shape)
        return self

    def gen_stars(self, n):
        nx, ny = self.shape
        x = np.random.uniform(0, nx, n)
        y = np.random.uniform(0, ny, n)
        flux = np.random.uniform(0, 100000., n)
        stars = np.rec.fromarrays((x, y, flux), names=['x', 'y', 'flux'])
        return stars

    def distort(self, model):
        if not hasattr(self, 'true_data') or self.true_data is None:
            self.true_data = self.orig_data.copy()
        self.distorted_data = model.apply(self.true_data)
        return self

    def plot(self, vmin=-100., vmax=100.):
        _, axes = pl.subplots(nrows=1, ncols=2, figsize=(12, 8),
                              sharex=True, sharey=True)
        axes[0].imshow(self.true_data)
        axes[0].set_xlabel('x [pixels]')
        axes[0].set_ylabel('y [pixels]')
        axes[0].set_title('true data (no distortion)')
        if self.distorted_data is not None:
            axes[1].imshow(self.distorted_data - self.true_data,
                           vmin=vmin, vmax=vmax)
            axes[1].set_xlabel('x [pixels]')
        axes[1].set_title('true - distorted')


# def line(size=1000, skylev=100, flux=1000, x_star=500.2, sigma=4., noise=True, plot=False):
#     """
#     """
#     x = np.arange(0, size, 1.)
#     y = np.full(size, skylev)
#     psf = np.exp(-0.5 * ((x-x_star)/sigma)**2) / np.sqrt(2 * np.pi) / sigma
#     y = y + flux * psf
#     if noise:
#         y = y + np.random.normal(loc=0., scale=np.sqrt(y), size=size)
    
#     y[-30:] = 0.
#     c = np.zeros_like(y)
#     s = np.zeros_like(y)
#     from_pocket = np.zeros_like(y)
#     to_pocket = np.zeros_like(y)
#     delta = np.zeros_like(y)
#     if plot:
#         pl.plot(x, y, 'k.')

#     return np.vstack((y, c, s, from_pocket, to_pocket, delta))


# def image(shape=(100,100), overscan=30, skylev=100., stars=None, model=None):
#     """
#     """
#     im = np.full(shape, skylev)

#     if noise:
#         im = im + np.random.normal(loc=0., scale=np.sqrt(y), size=shape)

# #    if model:
# #        return model(im)
