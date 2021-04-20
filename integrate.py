import numpy as np
from scipy.linalg import lu_factor, lu_solve
import scipy.optimize

import numbers
from runge_kutta import (
    runge_kutta4,
    runge_kutta5,
)
from interpolate import CubicHermiteInterpolate


def integrate(rhs, times, y0, args=(), method=runge_kutta4):
    if isinstance(y0, numbers.Number):
        y0 = np.array([y0], dtype=np.float64)
    elif not isinstance(y0, np.ndarray):
        raise TypeError('y0 should be a number or ndarray')

    T = len(times)
    n = len(y0)
    y = np.empty((T, n), dtype=y0.dtype)
    y[0] = y0
    for i, t in enumerate(times[:-1]):
        t0 = times[i]
        t1 = times[i+1]
        y[i+1] = method(rhs, t0, t1-t0, y[i], args)
    return y

def interpolate(y, times, rhs, new_times, args=()):
    n = y.shape[1]
    y_out = np.empty((len(new_times), n), dtype=y.dtype)

    # this is the index into new_times
    j = 0
    for i, t in enumerate(times[:-1]):
        t0 = times[i]
        t1 = times[i+1]
        new_t0 = new_times[j]

        if new_t0 > t1:
            continue

        # construct interpolator
        y_interp = CubicHermiteInterpolate(
            t0, t1, y[i], y[i+1],
            rhs(t0, y[i], *args), rhs(t1, y[i+1], *args)
        )

        # interpolate
        while new_t0 <= t1:
            y_out[j] = y_interp(new_t0)
            j += 1

            if j == len(new_times):
                return y_out

            new_t0 = new_times[j]

    return y_out

def adapt(error, times, thresh, alpha=0.01):
    # if the error is above tolerance:
    # - split those segments that have an error > thresh
    # - join segments whose combined error < alpha * thresh
    split = np.abs(error) > thresh
    join = (np.abs(error[:-1]) + np.abs(error[1:])) < alpha * thresh
    T = len(times) + np.sum(split)
    new_times = np.empty(T, dtype=times.dtype)
    index = 0
    i = 0
    while i < len(split):
        new_times[index] = times[i]
        index += 1
        if split[i]:
            new_times[index] = 0.5*(times[i] + times[i+1])
            index += 1
        elif i < len(join) and join[i]:
            i += 1
        i += 1

    new_times[index] = times[-1]
    return np.resize(new_times, index+1)

class MinimiseTraditional:
    def __init__(self, rhs, jac, drhs_dp, funcional,
                 dfunc_dy, ftimes, y0, rtol=1e-4, atol=1e-6):
        self.ftimes = ftimes
        self.functional = funcional
        self.dfunc_dy = dfunc_dy
        self.times = np.linspace(ftimes[0], ftimes[-1], 10)
        self.rhs = rhs
        self.drhs_dp = drhs_dp
        self.jac = jac
        self.y0 = y0
        self.rtol = rtol
        self.atol = atol

        self.total_error = []
        self.ntimes = []

    def __call__(self, p):

        # integrate ode
        sol = scipy.integrate.solve_ivp(
            self.rhs, (self.ftimes[0], self.ftimes[-1]),
            self.y0, args=(p,), rtol=self.rtol, atol=self.atol
        )

        # calculate error and sensitivities
        total_error, error, f, g = adjoint_error_and_sensitivities(
            self.rhs, self.jac, self.drhs_dp, self.functional,
            self.dfunc_dy, self.ftimes, sol.t,
            sol.y.T, args=(p,)
        )

        self.total_error.append(total_error)
        self.ntimes.append(len(self.times))

        return f, g

class MinimiseTraditionalNoGradient:
    def __init__(self, rhs, jac, drhs_dp, funcional,
                 dfunc_dy, ftimes, y0, rtol=1e-4, atol=1e-6):
        self.ftimes = ftimes
        self.functional = funcional
        self.dfunc_dy = dfunc_dy
        self.times = np.linspace(ftimes[0], ftimes[-1], 10)
        self.rhs = rhs
        self.drhs_dp = drhs_dp
        self.jac = jac
        self.y0 = y0
        self.rtol = rtol
        self.atol = atol

    def __call__(self, p):

        # integrate ode
        sol = scipy.integrate.solve_ivp(
            self.rhs, (self.ftimes[0], self.ftimes[-1]),
            self.y0, t_eval=self.ftimes, args=(p,),
            rtol=self.rtol, atol=self.atol
        )

        f = self.functional(sol.y.T)

        return f


class Minimise:
    def __init__(self, rhs, jac, drhs_dp, funcional,
                 dfunc_dy, ftimes, y0, rtol=1e-4, atol=1e-6):
        self.ftimes = ftimes
        self.functional = funcional
        self.dfunc_dy = dfunc_dy
        self.times = np.linspace(ftimes[0], ftimes[-1], 10)
        self.rhs = rhs
        self.drhs_dp = drhs_dp
        self.jac = jac
        self.y0 = y0
        self.rtol = rtol
        self.atol = atol

        self.total_error = []
        self.ntimes = []

    def __call__(self, p):
        # integrate ode
        while True:
            y = integrate(self.rhs, self.times, self.y0,
                          args=(p,), method=runge_kutta4)

            if not np.isinf(y).any():
                break

            self.times = adapt(
                np.full(len(self.times) - 1, np.inf),
                self.times, 1
            )

        # calculate error and sensitivities
        total_error, error, f, g = adjoint_error_and_sensitivities(
            self.rhs, self.jac, self.drhs_dp, self.functional,
            self.dfunc_dy, self.ftimes, self.times,
            y, args=(p,)
        )

        # adapt
        self.times = adapt(
            error, self.times,
            (f * self.rtol + self.atol) / len(self.times)
        )

        self.total_error.append(total_error)
        self.ntimes.append(len(self.times))

        return f, g



def integrate_adaptive(
        rhs, jac, functional, dfunc_dy, ftimes, y0, times=None, rtol=1e-6, atol=1e-6
):
    if isinstance(y0, numbers.Number):
        y0 = np.array([y0], dtype=np.float64)
    elif not isinstance(y0, np.ndarray):
        raise TypeError('y0 should be a number or ndarray')

    t_rhs_calls = 0
    t_jac_calls = 0
    f = 1
    total_error = 2 * (f * rtol + atol)
    if times is None:
        times = ftimes.copy()
        times = np.linspace(ftimes[0], ftimes[-1], 10)

    while True:
        T = len(times)

        print('integrate using times')
        print(times)
        y = integrate(rhs, times, y0, method=runge_kutta4)
        t_rhs_calls += T * 4

        total_error, error, f, rhs_calls, jac_calls = adjoint_error(
            rhs, jac, functional, dfunc_dy, ftimes, times, y
        )
        t_rhs_calls += rhs_calls
        t_jac_calls += jac_calls
        print('abs error {}'.format(total_error))
        print('rel error {}'.format(total_error/f))
        print(error)

        if abs(total_error) > f * rtol + atol:
            times = adapt(error, times, (f * rtol + atol) / T)
            T = len(times)
            print('new times')
            print(times)
        else:
            break

    return y, times, f, t_rhs_calls, t_jac_calls

def adjoint_error_and_sensitivities(
        rhs, jac, drhs_dp, functional, dfunc_dy, ftimes, times, y, args=(), method=runge_kutta5
):

    T = len(times)
    fT = len(ftimes)
    n = y.shape[1]

    # calculate the value and derivative of the function wrt y
    fy = interpolate(y, times, rhs, ftimes, args)
    Ju = dfunc_dy(fy)
    f = functional(fy)

    if Ju.shape[0] != fT:
        raise RuntimeError(
            'dfunc_dy (shape={}) should return length {}'
            .format(Ju.shape, fT)
        )

    # define the adjoint error equations
    def adjoint_error_and_sensitivities(
            t, phi_error_dJdp, y_interp, *args
    ):
        phi = phi_error_dJdp[:n]
        return np.concatenate((
            -jac(t, y_interp(t), phi, *args),
            (rhs(t, y_interp(t), *args) - y_interp.grad(t)) * phi,
            -drhs_dp(t, y_interp(t), phi, *args),
        ))

    phi = np.zeros(n, dtype=y.dtype)
    n_params = len(drhs_dp(times[-1], y[-1], phi, *args))
    phi_error_dJdp = np.zeros(n + 1 + n_params, dtype=y.dtype)
    error = np.empty(T-1, dtype=y.dtype)

    # this is index into ftimes
    j = len(ftimes) - 1
    for i in reversed(range(len(times)-1)):

        t0 = times[i]
        t1 = times[i+1]
        ft = ftimes[j]
        t = t1
        phi_error1 = phi_error_dJdp[-1-n_params]

        y_interp = CubicHermiteInterpolate(
            t0, t1, y[i], y[i+1],
            rhs(t0, y[i], *args), rhs(t1, y[i+1], *args)
        )

        # ft could be == to t1
        if ft >= t:
            phi_error_dJdp[:n] += Ju[j]
            j -= 1
            ft = ftimes[j]

        # break segment according to location of ftimes
        while ft > t0:
            # integrate
            phi_error_dJdp = method(adjoint_error_and_sensitivities,
                               t, ft-t,
                               phi_error_dJdp, (y_interp,) + args)

            # integrate over delta function
            phi_error_dJdp[:n] += Ju[j]

            # go to new time point
            j -= 1
            t = ft
            ft = ftimes[j]

        # ft is now <= to t0, integrate to t0
        phi_error_dJdp = method(adjoint_error_and_sensitivities,
                                t, t0-t,
                                phi_error_dJdp, (y_interp,) + args)
        error[i] = phi_error_dJdp[-1-n_params] - phi_error1


    return phi_error_dJdp[-1-n_params], error, \
        f, phi_error_dJdp[-n_params:]



def adjoint_error(
        rhs, jac, functional, dfunc_dy, ftimes, times, y, method=runge_kutta5
):

    T = len(times)
    fT = len(ftimes)
    n = y.shape[1]
    rhs_calls = 0
    method_calls = 0

    # calculate the value and derivative of the function wrt y
    fy = interpolate(y, times, rhs, ftimes)
    Ju = dfunc_dy(fy)
    f = functional(fy)

    if Ju.shape[0] != fT:
        raise RuntimeError(
            'dfunc_dy (shape={}) should return length {}'
            .format(Ju.shape, fT)
        )

    # define the adjoint error equations
    def adjoint_error(t, phi_error, y_interp):
        phi = phi_error[:n]
        return np.concatenate((
            -jac(t, y_interp(t), phi),
            (rhs(t, y_interp(t)) - y_interp.grad(t)) * phi,
        ))

    phi_error = np.zeros(n + 1, dtype=y.dtype)
    error = np.empty(T-1, dtype=y.dtype)

    # this is index into ftimes
    j = len(ftimes) - 1
    for i in reversed(range(len(times)-1)):

        t0 = times[i]
        t1 = times[i+1]
        ft = ftimes[j]
        t = t1
        phi_error1 = phi_error[-1]

        y_interp = CubicHermiteInterpolate(
            t0, t1, y[i], y[i+1], rhs(t0, y[i]), rhs(t1, y[i+1])
        )
        rhs_calls += 2

        # ft could be == to t1
        if ft >= t:
            phi_error[:n] += Ju[j]
            j -= 1
            ft = ftimes[j]

        # break segment according to location of ftimes
        while ft > t0:
            # integrate
            phi_error = method(adjoint_error,
                               t, ft-t,
                               phi_error, (y_interp,))
            method_calls += 1

            # integrate over delta function
            phi_error[:n] += Ju[j]

            # go to new time point
            j -= 1
            t = ft
            ft = ftimes[j]

        # ft is now <= to t0, integrate to t0
        phi_error = method(adjoint_error,
                           t, t0-t,
                           phi_error, (y_interp,))
        method_calls += 1
        error[i] = phi_error[-1] - phi_error1

    if method == runge_kutta5:
        method_to_func_calls = 6
    else:
        method_to_func_calls = np.nan

    rhs_calls += method_to_func_calls * method_calls
    jac_calls = method_to_func_calls * method_calls

    return phi_error[-1], error, f, rhs_calls, jac_calls


def adjoint_sensitivities_single_times(
        rhs, jac, drhs_dp, dfunc_dy, times, y, method=runge_kutta5
):
    T = len(times)
    if T != y.shape[0]:
        raise RuntimeError(
            'y (shape={}) should have length {}'
            .format(y.shape, T)
        )
    n = y.shape[1]

    def adjoint_sensitivities(t, phi_dJdp, y_interp):
        phi = phi_dJdp[:n]
        return np.concatenate((
            -jac(t, y_interp(t), phi),
            -drhs_dp(t, y_interp(t), phi),
        ))

    Ju = dfunc_dy(y)
    if Ju.shape[0] != T:
        raise RuntimeError(
            'dfunc_dy (shape={}) should return length {}'
            .format(Ju.shape, T)
        )
    phi = np.zeros(n, dtype=y.dtype)
    n_params = len(drhs_dp(times[-1], y[-1], phi))
    phi_dJdp = np.zeros(n + n_params, dtype=y.dtype)
    for i in reversed(range(len(times)-1)):
        # integrate over delta function
        phi_dJdp[:n] += Ju[i+1]

        t0 = times[i]
        t1 = times[i+1]

        y_interp = CubicHermiteInterpolate(
            t0, t1, y[i], y[i+1], rhs(t0, y[i]), rhs(t1, y[i+1])
        )

        # integrate
        phi_dJdp = method(adjoint_sensitivities,
                          t1, t0-t1, phi_dJdp, (y_interp,))

    return phi_dJdp[n:]


def adjoint_error_single_times(
        rhs, jac, dfunc_dy, times, y, method=runge_kutta5
):
    T = len(times)
    if T != y.shape[0]:
        raise RuntimeError(
            'y (shape={}) should have length {}'
            .format(y.shape, T)
        )
    n = y.shape[1]

    def adjoint_error(t, phi_error, y_interp):
        phi = phi_error[:n]
        return np.concatenate((
            -jac(t, y_interp(t), phi),
            (rhs(t, y_interp(t)) - y_interp.grad(t)) * phi,
        ))

    Ju = dfunc_dy(y)
    if Ju.shape[0] != T:
        raise RuntimeError(
            'dfunc_dy (shape={}) should return length {}'
            .format(Ju.shape, T)
        )
    phi_error = np.zeros(n + 1, dtype=y.dtype)
    error = np.empty(T-1, dtype=y.dtype)
    for i in reversed(range(len(times)-1)):
        # integrate over delta function
        phi_error[:n] += Ju[i+1]

        t0 = times[i]
        t1 = times[i+1]

        y_interp = CubicHermiteInterpolate(
            t0, t1, y[i], y[i+1], rhs(t0, y[i]), rhs(t1, y[i+1])
        )

        # integrate
        new_phi_error = method(adjoint_error,
                               t1, t0-t1, phi_error, (y_interp,))
        error[i] = new_phi_error[-1] - phi_error[-1]
        phi_error = new_phi_error

    return phi_error[-1], error

def adjoint_sensitivities_error(
        rhs, jac, drhs_dp, dfunc_dy, times, y, method=runge_kutta5
):
    T = len(times)
    if T != y.shape[0]:
        raise RuntimeError(
            'y (shape={}) should have length {}'
            .format(y.shape, T)
        )
    n = y.shape[1]

    def adjoint_sensitivities(t, phi_dJdp, y_interp):
        phi = phi_dJdp[:n]
        return np.concatenate((
            -jac(t, y_interp(t), phi),
            -drhs_dp(t, y_interp(t), phi),
        ))

    rhs_s = adjoint_sensitivities

    Ju = dfunc_dy(y)
    if Ju.shape[0] != T:
        raise RuntimeError(
            'dfunc_dy (shape={}) should return length {}'
            .format(Ju.shape, T)
        )
    phi = np.zeros(n, dtype=y.dtype)
    n_params = len(drhs_dp(times[-1], y[-1], phi))
    phi_dJdp = np.zeros(n + n_params, dtype=y.dtype)
    for i in reversed(range(len(times)-1)):
        # integrate over delta function
        phi_dJdp[:n] += Ju[i+1]

        t0 = times[i]
        t1 = times[i+1]

        y_interp = CubicHermiteInterpolate(
            t0, t1, y[i], y[i+1], rhs(t0, y[i]), rhs(t1, y[i+1])
        )

        # integrate
        phi_dJdp = method(adjoint_sensitivities,
                          t1, t0-t1, phi_dJdp, (y_interp,))

    return phi_dJdp[n:]


