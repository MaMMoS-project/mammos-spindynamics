"""Postprocessing functions."""

from scipy.optimize import least_squares
import mammos_entity as me
import mammos_units as u
import numpy as np
import warnings


def kuzmin(
    Ms_data,
    Ms_0,
    K1_0,
    T=None,
):
    """Evaluate micromagnetic intrinsic properties.

    If temperature T is given, evaluate them at that temperature.
    Otherwise, the three outputs `Ms`, `A`, and `K1` are going to be
    functions of temperature.

    :param Ms_data: Interpolator on spontaneous magnetisation data.
    :type Ms_data: scipy.interpolate.iterp1d
    :param Ms_0: Spontaneous magnetisation at temperature 0K.
    :type Ms_0: mammos_entity.Ms
    :param K1_0: Magnetocrystalline anisotropy at temperature 0K.
    :type K1_0: mammos_entity.Ku
    :param T: Temperature, defaults to None
    :type T: mammos_units.Quantity | int | float, optional
    :raises ValueError: Wrong unit.
    :return: Either functions of temperature or intrinsic properties
        at a given temperature.
    :rtype: (mammos_entity.Ms, mammos_entity.A, mammos_entity.K) | (callable)
    """
    if (
        Ms_0.unit != u.A / u.m
        or K1_0.unit != u.J / u.m**3
    ):
        raise ValueError("Wrong unit.")  # TODO add more ?

    def M_kuzmin(T_, T_c_, s_):
        return np.where(
            T_ < T_c_,
            Ms_0.value
            * (
                (1 - s_ * (T_ / T_c_) ** 1.5 - (1 - s_) * (T_ / T_c_) ** 2.5)
                ** (1.0 / 3)
            ),
            0.0,
        )

    def residuals(params_, T_, M_):
        T_c_, s_ = params_
        return M_ - M_kuzmin(T_, T_c_, s_)

    with warnings.catch_warnings(action="ignore"):
        results = least_squares(
            residuals,
            (400, 0.5),
            args=(Ms_data.x, Ms_data.y),
            bounds=((0, 0), (np.inf, np.inf)),
            jac="3-point",
        )
    T_c, s = results.x
    T_c = T_c * u.K
    D = (
        0.1509
        * ((6 * u.constants.muB) / (s * Ms_0)) ** (2.0 / 3)
        * u.constants.k_B
        * T_c
    ).si
    A_0 = me.A(Ms_0 * D / (4 * u.constants.muB), unit=u.J / u.m)

    def M_func(Temp):
        return me.Ms(M_kuzmin(Temp.value, T_c.value, s))

    def A_func(Temp):
        return me.A(A_0 * (M_func(Temp) / Ms_0) ** 2)

    def K_func(Temp):
        return me.Ku(K1_0 * (M_func(Temp) / Ms_0) ** 3)

    if T is not None:
        if not isinstance(T, u.Quantity) or T.unit != u.K:
            T = T * u.K
        Ms = me.Ms(M_func(T))
        A = me.A(A_func(T))
        Ku = me.Ku(K_func(T))
        return Ms, A, Ku
    else:
        return M_func, A_func, K_func
    return