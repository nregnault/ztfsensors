#!/usr/bin/env python3

from importlib.resources import files

import numpy as np

from scipy import sparse
from sksparse import cholmod

#from ruamel.yaml import YAML

import warnings
try:
    import jax
    import jax.numpy as np
    _HAS_JAX = True
except ImportError:
    warnings.warn("jax is not installed. using numpy instead")
    _HAS_JAX = False
    
import numpy as np


from ._pocket import PocketModel as PocketModelCPP

# ============= #
#  Internal Jax #
# ============= #
def _fill(pocket_q, pixel_q, cmax, nmax, alpha, beta):
    """ pocket charge filling function

    Parameters
    ----------
    pocket_q: float, Array
        charge in the pocket prior read-out

    pixel_q: float, Array
        pixel charge prior read-out (undistorted)

    cmax: float
        pocket capacity

    nmax: float
        pixel capacity (not quite the full well)

    alpha: float
        from pocket transfer dynamics

    beta: float
        to-pocket transfer dynamics

    Returns
    -------
    float, Array
        charge entering the pocket.
    """
    x = pocket_q / cmax
    y = pixel_q / nmax
    outval = cmax * (1 - x)**alpha * y**beta
    #    return np.clip(outval,  0.,  pixel_q)
    return outval

def _flush(pocket_q, cmax, alpha):
    """ pocket charge flushing function

    Parameters
    ----------
    pocket_q: float, Array
        charge in the pocket prior read-out

    cmax: float
        pocket capacity

    alpha: float
        from pocket transfer dynamics

    Returns
    -------
    float, Array
        charge leaving the pocket.
    """
    x = pocket_q / cmax
    outval = cmax * x**alpha
    #return np.clip(outval, 0, pocket_q)
    return outval

class PocketModel():
    
    def __init__(self, alpha, cmax, beta, nmax):
        """ 
        cmax: float
            pocket capacity
        
        nmax: float
            pixel capacity (not quite the full well)

        alpha: float
            from pocket transfer dynamics

        beta: float
            to-pocket transfer dynamics        

        """
        self._alpha = alpha
        self._cmax = cmax
        self._beta = beta
        self._nmax = nmax
        
                
    def flush(self, pocket_q):
        """  transfer of electrons from the pocket to the pixels.

        Parameters
        ----------
        pocket_q: float, Array
            charge in the pocket prior read-out

        Returns
        -------
        float, Array
            charge leaving the pocket.
        """
        return _flush(pocket_q,
                      self._cmax, self._alpha)
    
    def fill(self, pocket_q, pixel_q):
        """ transfer of electrons entering the pocket

        Parameters
        ----------
        pocket_q: float, Array
            charge in the pocket prior read-out

        pixel_q: float, Array
            pixel charge prior read-out (undistorted)
     
        Returns
        -------
        float, Array
            charge entering the pocket.
        """
        return _fill(pocket_q, pixel_q,
                     self._cmax, self._nmax, self._alpha, self._beta)

    def get_delta(self, pocket_q, pixel_q):
        """ net pocket charge transfert 

        Parameters
        ----------
        pocket_q: float, Array
            charge in the pocket prior read-out

        pixel_q: float, Array
            pixel charge prior read-out (undistorted)

        Returns
        -------
        float, Array
            pixel charge excess (>0) and deficit (<0) at the read-out.
        """
        from_pocket = self.flush(pocket_q)
        to_pocket = self.fill(pocket_q, pixel_q)
        delta = from_pocket - to_pocket
        return delta

    def get_pocket_and_corr(self, pocket_q, pixel_q):
        """ scanning function providing corrected pixel and new pocket charge
        
        Parameters
        ----------
        pocket_q: float, Array
            charge in the pocket prior read-out

        pixel_q: float, Array
            pixel charge prior read-out (undistorted)

        Returns
        -------
        list
            - new pocket charge: float, Array
            - corrected pixel: float, Array
        """
        delta = self.get_delta(pocket_q, pixel_q)
        pixel_corr = pixel_q + delta
        new_pocket = pocket_q - delta
        return new_pocket, pixel_corr
        
    def apply(self, pixels, init=None, backend="cpp"):
        """ pocket effect correction 
        
        Parameters
        ----------
        pixels: 2d-Array
            raw pixel map, including overhead of shape (M,N)

        init: None, Array
            initial condition of the pocket (M,). 
            If None, zero is assumed.

        backend: str
            backend used for the computation

        Returns
        -------
        2d-Array
            pocket effect on pixel map (M,N)
        """
        if backend == "jax":
            return self._scan_apply(pixels, init=init)
        
        elif backend == "numpy": 
            return self._forloop_apply(pixels, init=init)
        
        elif backend == "numpy-nr":
            return self._forloop_apply_baseline(pixels)
        
        elif backend == "cpp":
            thiscpp = PocketModelCPP(self._alpha, self._cmax, self._beta, self._nmax)
            return thiscpp.apply(pixels) # 0 is force here.
            
        else:
            raise ValueError(f"unknown backend {backend}")
        
    def _scan_apply(self, pixels, init=None):
        """ docstring, see: self.apply """
        # with for lax.scan | jax
        # atleast_2d and squeeze is to respect cpp-version behavior
        
        pixels = np.atleast_2d(pixels)
        if init is None:
            init = np.zeros(shape=pixels[:,0].shape)
        
        last_pocket, resbuff = jax.lax.scan(self.get_pocket_and_corr,
                                                init,
                                                pixels.T)
        return resbuff.T.squeeze()
    
    def _forloop_apply(self, pixels, init=None):
        """ docstring, see: self.apply """
        # with for loop | numpy
        pixels = np.atleast_2d(pixels)
        if init is None:
            pocket = np.zeros(shape=pixels[:,0].shape)
        else:
            pocket = init # for consistency between method

        resbuff = []
        for col in pixels.T:
            pocket, corr = self.get_pocket_and_corr(pocket, col)
            resbuff.append(corr) # build line by line
            
        return np.vstack(resbuff).T.squeeze()

    def _forloop_apply_baseline(self, pix):
        """apply the model to 2D image

        = original NR dev = 

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
            from_pocket = self.flush(pocket) # _flush -> flush
            to_pocket = self.fill(pocket, n_j) #  _fill -> fillflush
            delta = from_pocket - to_pocket
            output[:,j] = n_j + delta
            pocket -= delta
            # just making sure that the pocket contents never become negative
            #
            # we may clip silently, but it is better for now to know that the
            # correction is buggy and can throw the calculation into the ditch
            assert np.all(pocket >= 0.)

        return output


    # ============= #
    #  Properties   #
    # ============= #
    @property
    def backend(self):
        """ backend used for the apply() computation """
        return self._backend


    
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
    v0 = model.apply(pix, backend="cpp")
    for i in range(N):
        pix[i] += step
        vv = model.apply(pix, backend="cpp")
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


# === config 
import os
import pandas
import yaml
from yaml.loader import SafeLoader

_SOURCEDIR = os.path.dirname(os.path.realpath(__file__))
CORRECTION_FILEPATH = os.path.join(_SOURCEDIR, "data", "pocket_corrections.yaml")

# CONFIG
with open(CORRECTION_FILEPATH) as f:
    data = yaml.load(f, Loader=SafeLoader)
    POCKET_PARAMETERS = pandas.DataFrame(data["data"]).set_index(["ccdid", "qid"])


def get_config(ccdid, qid):
    """ returns the pocket effect parameter configuration for the given quadrant """
    return POCKET_PARAMETERS.loc[ccdid, qid]

