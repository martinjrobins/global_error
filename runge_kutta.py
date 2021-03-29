import numpy as np


def runge_kutta4(f, t0, h, y0, args):
    """classic 4th order method"""
    k = np.empty((4, len(y0)), dtype=y0.dtype)
    k[0] = f(t0, y0, *args)
    k[1] = f(t0 + 0.5*h, y0 + 0.5*h*k[0], *args)
    k[2] = f(t0 + 0.5*h, y0 + 0.5*h*k[1], *args)
    k[3] = f(t0 + h, y0 + h*k[2], *args)
    return y0 + h * (1/6*k[0] + 1/3*k[1] + 1/3*k[2] + 1/6*k[3])


def runge_kutta41(f, t0, h, y0, args):
    k = np.empty((4, len(y0)), dtype=y0.dtype)
    k[0] = f(t0, y0, *args)
    k[1] = f(t0 + 1/3*h, y0 + h*(1/3*k[0]), *args)
    k[2] = f(t0 + 2/3*h, y0 + h*(-1/3*k[0] + k[1]), *args)
    k[3] = f(t0 + h, y0 + h*(k[0] - k[1] + k[2]), *args)
    return y0 + h * (1/8*k[0] + 3/8*k[1] + 3/8*k[2] + 1/8*k[3])


def runge_kutta5(f, t0, h, y0, args):
    """dormond-price 5th order method"""
    k = np.empty((6, len(y0)), dtype=y0.dtype)
    k[0] = f(t0, y0, *args)
    k[1] = f(t0 + 1/5*h, y0 + h*(1/5*k[0]), *args)
    k[2] = f(t0 + 3/10*h, y0 + h*(3/40*k[0] + 9/40*k[1]), *args)
    k[3] = f(
        t0 + 4/5*h,
        y0 + h*(44/45*k[0] - 56/15*k[1] + 32/9*k[2]),
        *args
    )
    k[4] = f(
        t0 + 8/9*h,
        y0 + h*(19372/6561*k[0] - 25360/2187*k[1] + 64448/6561*k[2] -
                212/729*k[3]),
        *args
    )
    k[5] = f(
        t0 + h,
        y0 + h*(9017/3168*k[0] - 355/33*k[1] + 46732/5247*k[2] + 49/176*k[3] -
                5103/18656*k[4]),
        *args
    )

    return y0 + h*(35/384*k[0] + 500/1113*k[2] + 125/192*k[3] - 2187/6784*k[4] +
                   11/84*k[5])
