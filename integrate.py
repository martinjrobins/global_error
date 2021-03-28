import numpy as np
from scipy.linalg import lu_factor, lu_solve
from runge_kutta import (
    runge_kutta5,
    runge_kutta4,
    runge_kutta41,
    runge_kutta4_stage,
    runge_kutta41_stage,
    runge_kutta5_stage,
)
from interpolate import CubicHermiteInterpolate


def integrate(rhs, times, y0, args=()):
    y = np.empty_like(times)
    y[0] = y0
    for i, t in enumerate(times[:-1]):
        t0 = times[i]
        t1 = times[i+1]
        y[i+1] = runge_kutta4(rhs, t0, t1-t0, y[i], *args)
    return y


def integrate_adjoint(rhs, jac, times, y):
    n = y.shape[1]

    def adjoint(t, phi, y_interp):
        return -jac(y_interp(t), t, phi)

    phi = np.empty_like(times)
    phi[0] = y0
    for i, t in reversed(enumerate(times[:-1])):
        t0 = times[i]
        t1 = times[i+1]
        y = CubicHermiteInterpolate(
            t0, t1, y[i], y[i+1], rhs(y[i]), rhs(y[i+1])
        )
        phi[i] = runge_kutta5(adjoint, t1, t0-t1, phi[i+1], args=(y))
    return phi
