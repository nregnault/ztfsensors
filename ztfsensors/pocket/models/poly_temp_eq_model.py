from __future__ import annotations

import numpy as np
from scipy import sparse

from .base import BaseEquilibriumModel
from .factory import EquilibriumModelFactory


@EquilibriumModelFactory.register
class PolyTempEqModel(BaseEquilibriumModel):
    """
    Polynomial-based temperature-dependent equilibrium model.

    This model represents the equilibrium function as a piecewise polynomial
    in log-space (u = log(1 + x)) with temperature-dependent coefficients:

        f_eq(x, dT) = sum_{i} sum_{j} a_{i,j} * dT^j * u^i  (for u < u_knot)
                    + sum_{i} sum_{j} b_{i,j} * dT^j * (u - u_knot)^(i+2)  (for u >= u_knot)

    where dT = (T - temp_ref) / temp_scale and u = log(1 + x).

    The model has two regions:
    - Below x_knot: polynomial in u with coefficients p_deg
    - Above x_knot: additional polynomial in (u - u_knot) with coefficients q_deg

    Attributes
    ----------
    p_deg : list[int]
        Polynomial degrees in temperature for each power of u below the knot.
    q_deg : list[int]
        Polynomial degrees in temperature for each power of (u - u_knot) above the knot.
    x_knot : float
        Sky level value where the piecewise model transitions.
    u_knot : float
        Log-space knot position: log(1 + x_knot).
    temp_ref : float
        Reference temperature for normalization (Kelvin).
    temp_scale : float
        Temperature scale for normalization (Kelvin).
    """

    MODEL_NAME = "poly_temp_eq"
    MODEL_VERSION = "v1"
    HEADER_FIELDS = ("p_deg", "q_deg", "x_knot", "temp_ref", "temp_scale")

    def __init__(
        self,
        p_deg: list[int] | None = None,
        q_deg: list[int] | None = None,
        x_knot=200.0,
        temp_ref: float = 160.0,
        temp_scale: float = 1.0,
    ) -> None:
        """
        Initialize a polynomial temperature equilibrium model.

        Parameters
        ----------
        p_deg : list[int], optional
            List of temperature polynomial degrees for each u^i term.
            Default: [2, 2, 2] means quadratic in temp for u^0, u^1, u^2.
            Use None for a degree to skip that u power.
        q_deg : list[int], optional
            List of temperature polynomial degrees for each (u - u_knot)^(i+2) term.
            Default: [3, 3] means cubic in temp for (du)^2 and (du)^3.
        x_knot : float, optional
            Sky level value for the piecewise transition (default: 200.0).
        temp_ref : float, optional
            Reference temperature for normalization (default: 160.0 K).
        temp_scale : float, optional
            Temperature scale for normalization (default: 1.0 K).
        """
        self.p_deg = p_deg if p_deg is not None else [2, 2, 2]
        self.q_deg = q_deg if q_deg is not None else [3, 3]
        self.x_knot = x_knot
        self.u_knot = np.log(1.0 + x_knot)
        self.temp_ref = float(temp_ref)
        self.temp_scale = float(temp_scale)

    def build_design_matrix(self, x, ccd_temp) -> sparse.coo_matrix:
        """
        Build the design matrix for the polynomial temperature model.

        Constructs a sparse design matrix where each column corresponds to a
        coefficient in the model. The model is piecewise polynomial in log-space.

        Parameters
        ----------
        x : array_like
            Sky level values (must be > -1).
        ccd_temp : array_like
            CCD temperature values (must have same shape as x).

        Returns
        -------
        sparse.coo_matrix
            Sparse design matrix with shape (len(x), n_params).

        Raises
        ------
        ValueError
            If x contains values <= -1 or if x and ccd_temp have different shapes.
        """
        x = np.asarray(x)
        if np.any(x <= -1.0):
            raise ValueError("x must satisfy x > -1.")

        ccd_temp = np.asarray(ccd_temp)
        if x.shape != ccd_temp.shape:
            raise ValueError("x and ccd_temp must have same shape")

        u = np.log1p(x)
        dt = self.rescale_temp(ccd_temp)
        du = np.maximum(0.0, u - self.u_knot)

        # add simplification
        J = []
        for u_deg, t_deg in enumerate(self.p_deg):
            if t_deg is None:
                continue
            J.append((np.vander(dt, t_deg + 1).T * u**u_deg).T)

        for u_deg, t_deg in enumerate(self.q_deg):
            J.append((np.vander(dt, t_deg + 1).T * du ** (u_deg + 2)).T)

        return sparse.coo_matrix(np.hstack(J))

    @property
    def params_shape(self):
        """
        Shape of the parameter array.

        Returns
        -------
        tuple[int]
            1D shape tuple containing the total number of parameters.
            Computed as sum of (deg + 1) for all non-None degrees in p_deg and q_deg.
        """
        n_p = np.sum([d + 1 if d is not None else 0 for d in self.p_deg])
        n_q = np.sum([d + 1 if d is not None else 0 for d in self.q_deg])
        return (n_p + n_q,)

    # def as_dict(self):
    #     return {
    #         "model_name": self.MODEL_NAME,
    #         "model_version": self.MODEL_VERSION,
    #         "p_deg": self.p_deg,
    #         "q_deg": self.q_deg,
    #         "x_knot": self.x_knot,
    #         "temp_ref": self.temp_ref,
    #         "temp_scale": self.temp_scale,
    #         "mjd_interval_convention": "[start, end)",
    #     }


# class PolyTempEqModel(BaseEquilibriumModel):
#     """
#     Equilibrium model
#     """

#     MODEL_NAME = "poly_temp_eq"
#     MODEL_VERSION = "v1"

#     def __init__(
#         self,
#         p_deg: int = 3,
#         q2_deg: int = 3,
#         q3_deg: int = 3,
#         x_knot=200.0,
#         temp_ref: float = 160.0,
#         temp_scale: float = 1.0,
#     ) -> None:
#         self.p_deg = int(p_deg)
#         self.q2_deg = int(q2_deg)
#         self.q3_deg = int(q3_deg)
#         self.x_knot = x_knot
#         self.u_knot = np.log(1.0 + x_knot)
#         self.temp_ref = float(temp_ref)
#         self.temp_scale = float(temp_scale)


#     def build_design_matrix(self, x, ccd_temp) -> np.ndarray:
#         """ """
#         x = np.asarray(x)
#         if np.any(x <= -1.0):
#             raise ValueError("x must satisfy x > -1.")

#         u = np.log(1.0 + x)
#         ccd_temp = np.asarray(ccd_temp)

#         if x.shape != ccd_temp.shape:
#             raise ValueError("x and ccd_temp must have same shape")

#         dt = self.rescale_temp(ccd_temp)
#         du = np.maximum(0.0, u - self.u_knot)

#         J_p = (np.vander(dt, self.p_deg + 1).T * u).T
#         J_2 = (np.vander(dt, self.q2_deg + 1).T * du**2).T
#         J_3 = (np.vander(dt, self.q3_deg + 1).T * du**3).T

#         return sparse.coo_matrix(np.hstack([J_p, J_2, J_3]))

#     @property
#     def params_shape(self):
#         return ((self.p_deg + 1) + (self.q2_deg + 1) + (self.q3_deg + 1),)


#     def as_dict(self):
#         return {
#             "model_name": self.MODEL_NAME,
#             "model_version": self.MODEL_VERSION,
#             "p_deg": self.p_deg,
#             "q2_deg": self.q2_deg,
#             "q3_deg": self.q3_deg,
#             "x_knot": self.x_knot,
#             "temp_ref": self.temp_ref,
#             "temp_scale": self.temp_scale,
#             "mjd_interval_convention": "[start, end)",
#         }
