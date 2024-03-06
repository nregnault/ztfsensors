#!/usr/bin/env python3

import time
import logging
from importlib.resources import files

import numpy as np
from pandas.core.arrays.period import delta_to_tick
import pylab as pl
from scipy import sparse
from sksparse import cholmod

from ruamel.yaml import YAML

# this is the C++ version
from ._pocket import _PocketModel



class PocketModel:
    """A pure numpy version of the pocket effect model
    """
    def __init__(self, alpha, cmax, beta, nmax):
        """constructor - a set of
        """
        self.alpha = alpha
        self.cmax = cmax
        self.beta = beta
        self.nmax = nmax

    def _flush(self, q_j):
        """electrons flushed from the pocket when reading

        Parameters
        ----------
        q_i : array-like of floats
          pocket content, j-th column

        Returns
        -------
        number of electrons transfered from pocket : array-like of floats
        """
        x = q_j / self.cmax
        from_pocket = np.clip(self.cmax * np.pow(x, self.alpha), 0., q_j)
        return from_pocket

    def _fill(self, q_j, n_j):
        """electrons transfered from the pixel to the pocket

        Parameters
        ----------
        q_j : array-like of floats
          pocket contents j-th column
        n_j : array-like of floats
          pixel contents, j-th column

        """
        x = q_j / self.cmax
        y = n_j / self.nmax
        to_pocket = np.clip(self.cmax * np.pow(1.-x, self.alpha) * np.pow(y, self.beta),
                            0.,  n_j)

    def apply(self, pix):
        """apply the model to 2D image

        Parameters
        ----------
        pix : 2D array-like of floats
          we assume that i labels the rows and j labels the physical columns.

        .. note:: columns and rows are *not* interchangeable here !
        """
        nrows, ncols = pix.shape

        output = np.zeros_like(pix)
        pocket = np.zeros(nrows)

        for j in range(ncols):
            n_j = pix[:,j]
            from_pocket = self._flush(pocket)
            to_pocket = self._fill(pocket, n_j)
            delta = from_pocket - to_pocket
            output[:,j] = n_j + delta
            pocket -= delta
            # just making sure that the pocket contents never become negative
            #
            # we may clip silently, but it is better for now to know that the
            # correction is buggy and can throw the calculation into the ditch
            assert np.all(pocket >= 0.)

        return output



def pocket_model_derivatives(model, pix, step=0.01):
    """model derivatives w.r.t the pixel values

    For now, we use numerical derivatives. It is probably possible to do better.

    Parameters
    ----------
    model : PocketModel
      the pocket effect model
    pix : array_like
      the pixel array
    step : float
      numerical step

    Returns
    -------
    jacobian matrix : array_like
    """
    N = len(pix)
    J = np.zeros((N, N))
    v0 = model.apply(pix)
    for i in range(N):
        pix[i] += step
        vv = model.apply(pix)
        J[i] = (vv-v0)/step
        pix[i] -= step
    return J


def correct_1d(model, pix, step=0.01, n_iter=5):
    """Reconstruct the undistorted pixel values (1D version)

    The undistorted pixel values are actually parameters of a least-square fit. At each step,
    we solve the normal equation:

    .. math::
        J^T W J (pix) = J^T W (pix-model)

    where :math:`J` is the jacobian matrix of the model, :math:`W` is a weight
    matrix and :math:`pix-model` are the residuals of the previous iteration.

    To speed things up, the Hessian factorization is not recomputed and is
    recycled at each step.

    Parameters
    ----------
    model : PocketModel
      the model used in the reconstruction
    pix : array_like
      the raw pixel array. The raw pixel array is expected to include the overscan.
      the overscan width (30 pixels) is currently hardcoded.
    step : float
      derivative step
    n_iter : int
      number of iterations

    Returns
    -------
    undistorted pixel array : array_like

    """
    default_pix_val = np.median(pix)

    J = pocket_model_derivatives(model, pix) # was 'sky'
    i,j = np.meshgrid(np.arange(J.shape[0]), np.arange(J.shape[1]))
    v = J[i.flatten(), j.flatten()]
    idx = np.abs(v)>1.E-4 # was 1.E-5
    i,j = i.flatten(), j.flatten()
    JJ = sparse.coo_matrix((v[idx], (i[idx], j[idx])), shape=J.shape)
    H = JJ.T @ JJ
    f = cholmod.cholesky(H) # , ordering_method='metis')

    current_state = pix.copy()
    current_state[-30:] = 0.
    current_state[0:3] = default_pix_val
    delta_tot = np.zeros_like(current_state)
    start = time.perf_counter()
    mask = np.zeros_like(current_state).astype(int)

    for i in range(n_iter):
        res = pix - model.apply(current_state)
        delta = f.solve_LDLt(JJ.T @ res)
        delta_tot += delta
        current_state += delta
        # we need to force the overscan to zero
        current_state[-30:] = 0.
        if i == 0:
            current_state[:2] = default_pix_val
        mask[current_state<0] = 1
        current_state[current_state<0] = default_pix_val
    stop = time.perf_counter()
    logging.info(f'time: {stop-start}')

    return current_state, delta_tot, mask


def correct_2d(model, pix, step=0.01, n_iter=4):
    """Reconstruct the undistorted pixel values (2D version)

    The undistorted pixel values are actually parameters of a least-square fit. At each step,
    we solve the normal equation:

    .. math::
        J^T W J (pix) = J^T W (pix-model)

    where :math:`J` is the jacobian matrix of the model, :math:`W` is a weight
    matrix and :math:`pix-model` are the residuals of the previous iteration.

    To speed things up, the Hessian factorization is not recomputed and is
    recycled at each step.

    Parameters
    ----------
    model : PocketModel
      the model used in the reconstruction
    pix : array_like
      the raw pixel array. The raw pixel array is expected to include the overscan.
      the overscan width (30 pixels) is currently hardcoded.
    step : float
      derivative step
    n_iter : int
      number of iterations

    Returns
    -------
    undistorted pixel array : array_like

    """
    default_pix_val = np.median(pix)

    line_prof = np.full(pix.shape[0], default_pix_val)
    print(line_prof)
    J = pocket_model_derivatives(model, line_prof) # was 'sky'
    i,j = np.meshgrid(np.arange(J.shape[0]), np.arange(J.shape[1]))
    v = J[i.flatten(), j.flatten()]
    idx = np.abs(v)>1.E-5
    i,j = i.flatten(), j.flatten()
    JJ = sparse.coo_matrix((v[idx], (i[idx], j[idx])), shape=J.shape)
    H = JJ.T @ JJ
    f = cholmod.cholesky(H, ordering_method='best')

    current_state = pix.copy()
    current_state[:,-30:] = 0.
    current_state[0:2] = default_pix_val
    delta_tot = np.zeros_like(current_state)
    mask = np.zeros_like(current_state).astype(int)
    start = time.perf_counter()
    for i in range(n_iter):
        res = pix - model.apply(current_state)
        delta = f.solve_LDLt(JJ.T @ res)
        # delta = f(JJ.T @ res)
        delta_tot += delta
        current_state += delta
        current_state[:,-30:] = 0.
        mask[current_state<0] = 1
        # current_state[:,:3] = default_pix_val
        current_state[current_state<0.] = default_pix_val
    stop = time.perf_counter()
    print(f'time: {stop-start}')

    return current_state, delta_tot, mask


def get_model_parameter_file(filename=None):
    """
    """
    if filename is None:
        filename = 'pocket_corrections.yaml'
    return files(__package__).joinpath('data', filename)


class PocketModelServer:

    def __init__(self, filename=None):
        """Constructor
        """
        path = get_model_parameter_file(filename)
        yaml = YAML()
        self.db = yaml.load(path)
        self.data = self.db['data']

    def __call__(self, ccdid, qid, mjd=None):
        """
        """
