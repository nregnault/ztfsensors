#!/usr/bin/env python3

import time

import numpy as np
import pylab as pl
from scipy import sparse
from sksparse import cholmod

import _pocket


def deriv(model, pix, step=0.01):
    """model derivatives
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


def fit(model, pix, step=0.01):
    """A naive fit method.
    """
    # first, create a distorted image and display it
    distorted = model.apply(pix)
    fig, axes = pl.subplots(nrows=2, ncols=2, figsize=(12,12), sharex=True)
    axes[0,0].plot(pix, 'k.', label='original')
    axes[0,0].plot(distorted, 'r+:', label='distorted')
    axes[0,0].set_title('original & distorted')
    axes[0,0].legend()

    # that, try to reconstruct the original pixels from the distorted image
    # our starting point is precisely the distorted image
    default_pix_val = np.median(pix)
    start = time.perf_counter()
    current_state = distorted.copy()
    current_state[-30:] = 0.
    current_state[0] = default_pix_val
    delta_tot = np.zeros_like(current_state)
    for i in range(4):
        res = distorted - model.apply(current_state)
        J = deriv(model, current_state)
        H = J.T @ J
        delta = np.linalg.inv(H) @ (J.T @ res)
        delta_tot += delta
        current_state += delta
        current_state[-30:] = 0.
        current_state[0:2] = default_pix_val
    stop = time.perf_counter()
    print(f'time: {stop-start}')

    axes[0,1].plot(delta_tot, 'b.')
    axes[0,1].set_title('correction')
    res = distorted - model.apply(current_state)
    axes[1,0].plot(res, 'b.')
    axes[1,0].set_title('residuals')
    axes[1,0].set_xlabel('pix')

    axes[1,1].plot(pix, 'k.', label='original')
    axes[1,1].plot(current_state, 'c+:', label='reconstructed')
    axes[1,1].sharey(axes[0,0])
    axes[1,1].set_ylim((-20, 360.))
    axes[1,1].legend()
    axes[1,1].set_title('original & reconstructed')
    axes[1,1].set_xlabel('pix')


def fast_fit(model, pix, sky, step=0.01):
    """A tentatively faster reconstruction 

       We can attempt to speed up the reconstruction by (1) recycling the
       matrix and its factorization (2) dropping the smaller derivatives to
       make it more sparse (2) re-implementing everything in C++
    """
    # first, create a distorted image and display it
    distorted = model.apply(pix)
    fig, axes = pl.subplots(nrows=2, ncols=2, figsize=(12,12), sharex=True)
    axes[0,0].plot(pix, 'k.', label='original')
    axes[0,0].plot(distorted, 'r+:', label='distorted')
    axes[0,0].set_title('original & distorted')
    axes[0,0].legend()

    default_pix_val = np.median(pix)

    J = deriv(model, sky)
    i,j = np.meshgrid(np.arange(J.shape[0]), np.arange(J.shape[1]))
    v = J[i.flatten(), j.flatten()]
    idx = np.abs(v)>1.E-5
    i,j = i.flatten(), j.flatten()
    JJ = sparse.coo_matrix((v[idx], (i[idx], j[idx])), shape=J.shape)
    H = JJ.T @ JJ
    f = cholmod.cholesky(H) # , ordering_method='metis')
    # return H, v, idx
    # H = J.T @ J
    # H_inv = np.linalg.inv(H)

    # that, try to reconstruct the original pixels from the distorted image
    # our starting point is precisely the distorted image
    current_state = distorted.copy()
    current_state[-30:] = 0.
    current_state[0:2] = default_pix_val
    delta_tot = np.zeros_like(current_state)
    start = time.perf_counter()
    for i in range(4):
        # s = time.perf_counter()
        res = distorted - model.apply(current_state)
        # print(f'model eval: {time.perf_counter() - s}')
        # s = time.perf_counter()
        # delta = H_inv @ J.T @ res
        delta = f.solve_LDLt(JJ.T @ res)
        # print(f'inv: {time.perf_counter() - s}')
        delta_tot += delta
        current_state += delta
        current_state[-30:] = 0.
        # current_state[0:2] = default_pix_val
    stop = time.perf_counter()
    print(f'time: {stop-start}')

    axes[0,1].plot(delta_tot, 'b.')
    axes[0,1].set_title('correction')
    axes[1,0].plot(res, 'b.')
    axes[1,0].set_title('residuals')

    axes[1,1].plot(pix, 'k.', label='original')
    axes[1,1].plot(current_state, 'c+:', label='reconstructed')
    axes[1,1].sharey(axes[0,0])
    axes[1,1].set_ylim((-20, 360.))
    axes[1,1].legend()
    axes[1,1].set_title('original & reconstructed')

    return H, JJ
