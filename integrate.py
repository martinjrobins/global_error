import numpy as np
from .runge_kutta import (
    runge_kutta5,
    runge_kutta4,
    runge_kutta41,
    runge_kutta4_stage,
    runge_kutta41_stage,
    runge_kutta5_stage,
)


def integrate(rhs, times, y0):
    y = np.empty_like(times)
    y[0] = y0
    for i, t in enumerate(times[:-1]):
        y[i+1] = \
            runge_kutta5(rhs, times[i], times[i+1] - times[i], y[i])
    return y

class Interpolate:
    def __init__(self, rhs, t0, h, y0):
        self.y, _ = runge_kutta4_stage(rhs, t0, h, y0)
        # fit polynomial to y
    def __call__(self, t):
        # return polynomial(t-t0)


def integrate_adjoint(rhs, drhs_dy, times, y):
    n = y.shape[1]

    def adjoint_rhs(t, aug_y, y_interp):
        y = aug_y[:n]
        phi = aug_y[n:]
        return -drhs_dy(y_interp(t), t) * phi

    aug_y = np.empty_like(times)
    aug_y[0] = y0
    for i, t in reverse(enumerate(times[:-1])):
        t0 = times[i]
        t1 = times[i+1]
        y_interp = Interpolate(rhs, t0, t1-t0, y[i])
        aug_y[i] = runge_kutta5(adjoint_rhs, t1, t1-t0, aug_y[i+1])
    return y
