#!/usr/bin/env python3

import time
import logging

import numpy as np
from pandas.core.arrays.period import delta_to_tick
import pylab as pl
from scipy import sparse
from sksparse import cholmod

from ._pocket import PocketModel


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
