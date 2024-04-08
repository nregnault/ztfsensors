import numpy as np
from sksparse import cholmod

from .pocket import DEFAULT_BACKEND

def correct_pixels(model, pixels,
                    hessian=None,
                    n_overscan=30,
                    n_iter=4, backend=DEFAULT_BACKEND):
    """ top level method returning model-corrected pixels

    Parameters
    ----------
    pixels: 2d-array
        raw-pixel + overscan (N,M+overscan_size)
        pixels are expected to be corrected from
        non-linearity and overscan.

    hessian: scipy.sparse.Matrix, None
        sparse hessien matrix used to fit the model.

    n_overscan: int
        number of overscan columns. In the input pixels

    n_iter: int
        number of iteration for the fit.

    backend: string
        backend used to apply the model (see self.apply()

    Returns
    -------
    2d-array
        corrected raw pixels.
    """

    default_pixel_value = np.median(pixels)

    # build hessian if needed
    if hessian is None:
        test_column = np.full(pixels.shape[0], default_pixel_value )
        hessian = model.get_sparse_hessian(test_column, backend=backend)

    # Cholesky factorisation
    cholesky_f = cholmod.cholesky(hessian.tocsc(), ordering_method='best') # tocsc() to rm warnings

    # Actual iterative fit;
    current_state = pixels.copy()
    current_state[:, -n_overscan:] = 0 # constraints | overscan = no data
    current_state[0:2] = default_pixel_value # stability

    for i in range(n_iter):
        res = pixels - model.apply(current_state, backend=backend)
        delta = cholesky_f.solve_LDLt(hessian.T @ res)

        current_state += delta # get closer to the truth
        # reset constraints
        current_state[:,-n_overscan:] = 0. # force 0 at overscan
        current_state[current_state<0.] = default_pixel_value

    return current_state
