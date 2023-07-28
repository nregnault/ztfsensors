#!/usr/bin/env python

"""
Simulation of a test data sample
"""
import logging

import numpy as np
import pandas
import pylab as pl
import scipy.sparse
from astropy.time import Time
from croaks import DataProxy
from numba.experimental import jitclass
from saunerie.fitparameters import FitParameters
from scipy.optimize import least_squares

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', 
                    level=logging.INFO)


class PocketModel:
    """A model of the flow of charges trapped in the pocket and released with a delay.
    """
    def __init__(self, nx=1000):
        self.pars = self.init_pars(nx)
        self.data = np.zeros((2,nx))

    def init_pars(self, nx):
        """
        """
        fit_pars = FitParameters([('alpha', 1), ('cmax', 1), 
                                  ('beta', 1), ('nmax', 1), 
                                  ('ni', nx)])
        fit_pars['alpha'].full[:] = 1.74042342
        fit_pars['cmax'].full[:] = 1756.63622992
        fit_pars['beta'].full[:] = 0.39522657
        fit_pars['nmax'].full[:] = 251019.47431971

        return fit_pars

    def _fill(self, q_i, n_i):
        """pocket fill function.
        """
        q_i, n_i = np.array(q_i), np.array(n_i)
        alpha, c_max, beta, n_max = self.pars.full[:4]
        x = q_i / c_max
        f_n = np.power(n_i/n_max, beta)
        to_pocket = c_max * f_n * np.power(1-x, alpha)
        return to_pocket

    def _d_fill(self, q_i, n_i):
        """derivatives of pocket fill function
        """
        q_i, n_i = np.array(q_i), np.array(n_i)
        alpha, c_max, beta, n_max = self.pars.full[:4]

        x = q_i / c_max
        f_n = np.power(n_i/n_max, beta)
        d_f_n = beta * np.power(n_i/n_max, beta-1.) / n_max
        pp = np.power(1-x, alpha)
        dpp = -alpha * np.power(1-x, alpha-1.)
        dvaldn = c_max * d_f_n*pp
        dvaldc = f_n*dpp
        return np.array([dvaldc, dvaldn]).T

    def _flush(self, q_i):
        """
        """
        alpha, c_max = self.pars.full[:2]
        x = q_i / c_max
        from_pocket = c_max * np.power(x, alpha)
        return from_pocket

    def _d_flush(self, q_i):
        """
        """
        q_i = np.array(q_i)
        alpha, c_max = self.pars.full[:2]

        x = q_i / c_max 
        dval = np.array(alpha * np.power(x, alpha-1))
        z = np.zeros_like(dval)
        return np.array([dval, z]).T

    def pocket_charge(self, n_eq):
        """
        """
        alpha, c_max, beta, n_max = self.pars.full[:4]
        g_eq = np.power(n_eq/n_max, beta/alpha)
        return c_max * g_eq / (1. + g_eq)

    def apply(self, data, jac=False):
        """Apply the model to time series of simulated data
       
        This is the slow version of the model. Should be replaced by a compiled code. 
        """
        N = len(self.pars['ni'].full)
        assert N == data.shape[1]

        ret = np.zeros(N)
        dret = np.zeros(N)
        for i in range(1,N):
            n_i = data[0,i]
            q_ii = data[1,i-1]
            from_pocket = self._flush(q_ii)
            to_pocket = self._fill(q_ii, n_i)
            delta = from_pocket - to_pocket
            ret[i] = n_i + delta
            data[1,i] = q_ii - delta
            if jac:
                d_flush = self._d_flush(q_ii)
                d_fill = self._d_fill(q_ii, n_i)
                d_delta = d_flush - d_fill
                dret[i] = 1. + d_delta[1]
        if jac:
            
            J = scipy.sparse.coo_matrix((dret, (i, j)), shape=(N,np))
            return ret, dret
        return ret

    def __call__(self, p, jac=False):
        """
        """
        self.pars.free = p
        self.data[0,:] = self.pars['ni'].free
        if jac:
            ret, dret = self.apply(self.data, jac=True)
            return ret, dret
        return ret
