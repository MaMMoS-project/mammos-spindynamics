"""Postprocessing functions."""

from collections.abc import Callable
from functools import partial
import mammos_entity as me
import mammos_units as u
import numbers
import numpy as np
import pandas as pd
from scipy import interpolate, optimize
from typing import NamedTuple
import warnings


def kuzmin(
    Ms_data: pd.DataFrame,
    K1_0: me.Entity,
) -> tuple(Callable[[u.Quantity], me.Entity]):
    """Evaluate micromagnetic intrinsic properties.

    If temperature T is given, evaluate them at that temperature.
    Otherwise, the three outputs `Ms`, `A`, and `K1` are going to be
    functions of temperature.

    :param Ms_data: Interpolator on spontaneous magnetisation data.
    :type Ms_data: scipy.interpolate.iterp1d
    :param K1_0: Magnetocrystalline anisotropy at temperature 0K.
    :type K1_0: mammos_entity.Entity
    :raises ValueError: Wrong unit.
    :return: Either functions of temperature or intrinsic properties
        at a given temperature.
    :rtype: (mammos_entity.Ms, mammos_entity.A, mammos_entity.K) | (callable)
    """
    if not isinstance(K1_0, u.Quantity) or K1_0.unit != u.J / u.m**3:
        K1_0 = me.Ku(K1_0, unit=u.J / u.m**3)

    Ms_0 = me.Ms(Ms_data["M[A/m]"][0], unit=u.A / u.m)
    M_kuzmin = partial(generic_kuzmin, Ms_0.value)

    def residuals(params_, T_, M_):
        T_c_, s_ = params_
        return M_ - M_kuzmin(T_c_, s_, T_)

    with warnings.catch_warnings(action="ignore"):
        results = optimize.least_squares(
            residuals,
            (400, 0.5),
            args=(Ms_data["T[K]"], Ms_data["M[A/m]"]),
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

    return KuzminResult(
        Ms_function_of_temperature(Ms_0.value, T_c.value, s),
        A_function_of_temperature(A_0, Ms_0.value, T_c.value, s),
        K1_function_of_temperature(K1_0, Ms_0.value, T_c.value, s),
    )


def generic_kuzmin(Ms_0, T_c, s, T):
    return np.where(
        T < T_c,
        Ms_0 * ((1 - s * (T / T_c) ** 1.5 - (1 - s) * (T / T_c) ** 2.5) ** (1.0 / 3)),
        0.0,
    )


class A_function_of_temperature:
    def __init__(self, A_0, Ms_0, T_c, s):
        self.A_0 = A_0
        self.Ms_0 = Ms_0
        self.T_c = T_c
        self.s = s

    def __repr__(self):
        return "A(T)"

    def __call__(self, T: numbers.Real | u.Quantity):
        if isinstance(T, u.Quantity):
            T = T.to(u.K).value
        return me.A(
            self.A_0 * (generic_kuzmin(self.Ms_0, self.T_c, self.s, T) / self.Ms_0) ** 2
        )


class K1_function_of_temperature:
    def __init__(self, K1_0, Ms_0, T_c, s):
        self.K1_0 = K1_0
        self.Ms_0 = Ms_0
        self.T_c = T_c
        self.s = s

    def __repr__(self):
        return "K1(T)"

    def __call__(self, T: numbers.Real | u.Quantity):
        if isinstance(T, u.Quantity):
            T = T.to(u.K).value
        return me.Ku(
            self.K1_0
            * (generic_kuzmin(self.Ms_0, self.T_c, self.s, T) / self.Ms_0) ** 3
        )


class Ms_function_of_temperature:
    def __init__(self, Ms_0, T_c, s):
        self.Ms_0 = Ms_0
        self.T_c = T_c
        self.s = s

    def __repr__(self):
        return "Ms(T)"

    def __call__(self, T: numbers.Real | u.Quantity):
        if isinstance(T, u.Quantity):
            T = T.to(u.K).value
        return me.Ms(generic_kuzmin(self.Ms_0, self.T_c, self.s, T))


class KuzminResult(NamedTuple):
    Ms: Callable[[u.Quantity], me.Entity]
    A: Callable[[u.Quantity], me.Entity]
    K1: Callable[[u.Quantity], me.Entity]
