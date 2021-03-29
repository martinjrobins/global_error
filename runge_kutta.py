import jax.numpy as jnp
import numpy as np


def runge_kutta(f, t0, h, y0, tableau_c, tableau_b, tableau_a):
    s = len(tableau_b)
    k = np.empty(s, dtype=float)
    k[0] = f(t0, y0)
    dy = tableau_b[0] * k[0]
    for i in range(s-1):
        a_dot_k = tableau_a[i][0]*k[0]
        for j in range(1, i+1):
            a_dot_k += tableau_a[i][j]*k[j]
        k[i+1] = f(t0 + tableau_c[i]*h, y0 + h * a_dot_k)
        dy += tableau_b[i+1] * k[i+1]
    y1 = y0 + h * dy
    return y1

def runge_kutta4(f, t0, h, y0, args):
    """classic 4th order method"""
    k = np.empty((4, len(y0)), dtype=y0.dtype)
    k[0] = f(t0, y0, *args)
    k[1] = f(t0 + 0.5*h, y0 + 0.5*h*k[0], *args)
    k[2] = f(t0 + 0.5*h, y0 + 0.5*h*k[1], *args)
    k[3] = f(t0 +     h, y0 +     h*k[2], *args)
    return y0 + h * (1/6*k[0] + 1/3*k[1] + 1/3*k[2] + 1/6*k[3])

def runge_kutta4_stage(f, t0, h, y0):
    """classic 4th order method"""
    y = np.empty(4)
    x = np.empty(4)
    y[0] = y0
    x[0] = t0
    k = f(x[0], y[0])
    y[1] = y0 + 0.5*h*k
    x[1] = t0 + 0.5*h
    k = f(x[1], y[1])
    y[2] = y0 + 0.5*h*k
    x[2] = t0 + 0.5*h
    k = f(x[2], y[2])
    y[3] = y0 + h*k
    x[3] = t0 + h
    k = f(x[3], y[3])
    return y, x

def runge_kutta41(f, t0, h, y0, args):
    k = np.empty((4, len(y0)), dtype=y0.dtype)
    k[0] = f(t0, y0, *args)
    k[1] = f(t0 + 1/3*h, y0 + h*( 1/3*k[0]), *args)
    k[2] = f(t0 + 2/3*h, y0 + h*(-1/3*k[0] + k[1]), *args)
    k[3] = f(t0 +     h, y0 + h*(     k[0] - k[1] + k[2]), *args)
    return y0 + h * (1/8*k[0] + 3/8*k[1] + 3/8*k[2] + 1/8*k[3])

def runge_kutta41_stage(f, t0, h, y0):
    x = np.empty(4)
    k = np.empty(4)
    y = np.empty(4)
    y[0] = y0
    x[0] = t0
    k[0] = f(x[0], y[0])
    y[1] = y0 + h*(1/3*k[0])
    x[1] = t0 + 1/3*h
    k[1] = f(x[1], y[1])
    y[2] = y0 + h*(-1/3*k[0] + k[1])
    x[2] = t0 + 2/3*h
    k[2] = f(x[2], y[2])
    y[3] = y0 + h*(k[0] - k[1] + k[2])
    x[3] = t0 + h
    k[3] = f(x[3], y[3])
    y[3] = y0 + h * (1/8*k[0] + 3/8*k[1] + 3/8*k[2] + 1/8*k[3])
    return y, x

def runge_kutta5(f, t0, h, y0, args):
    """dormond-price 5th order method"""
    k = np.empty((6, len(y0)), dtype=y0.dtype)
    k[0] = f(t0, y0, *args)
    k[1] = f(t0 + 1/5*h, y0 + h*( 1/5*k[0]), *args)
    k[2] = f(t0 + 3/10*h,y0 + h*(3/40*k[0] + 9/40*k[1]), *args)
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

def runge_kutta5_stage(f, t0, h, y0, *args):
    """dormond-price 5th order method"""
    k = np.empty(6)
    y = np.empty(6)
    y[0] = y0
    k[0] = f(t0, y[0], *args)
    y[1] = y0 + h*( 1/5*k[0])
    k[1] = f(t0 + 1/5*h, y[1], *args)
    y[2] = y0 + h*(3/40*k[0] + 9/40*k[1])
    k[2] = f(t0 + 3/10*h, y[2], *args)
    y[3] = y0 + h*(44/45*k[0] - 56/15*k[1] + 32/9*k[2])
    k[3] = f(t0 + 4/5*h, y[3], *args)
    y[4] = y0 + h*(19372/6561*k[0] - 25360/2187*k[1] + 64448/6561*k[2] - 212/729*k[3])
    k[4] = f(t0 + 8/9*h, y[4], *args)
    y[5] = y0 + \
        h*(9017/3168*k[0] - 355/33*k[1] + 46732/5247*k[2] + 49/176*k[3] - 5103/18656*k[4])
    k[5] = f(t0 + h, y[5], *args)
    return y, k




def integrate_tableau(rhs, times, y0):
    tableau_a = ((1/3,), (-1/3, 1.0), (1.0, -1.0, 1.0))
    tableau_b = (1/8, 3/8, 3/8, 1/8)
    tableau_c = (1/3, 2/3, 1.0)
    y = np.empty_like(times)
    y[0] = y0
    for i, t in enumerate(times[:-1]):
        y[i+1] = runge_kutta(rhs, times[i], times[i+1] - times[i],
                             y[i], tableau_c, tableau_b, tableau_a)
    return y
