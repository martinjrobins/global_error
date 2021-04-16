import numpy as np
from scipy.linalg import lu_factor, lu_solve

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

def interpolate(y, times, rhs, new_times):
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
            t0, t1, y[i], y[i+1], rhs(t0, y[i]), rhs(t1, y[i+1])
        )

        # interpolate
        while new_t0 <= t1:
            y_out[j] = y_interp(new_t0)
            j += 1

            if j == len(new_times):
                return y_out

            new_t0 = new_times[j]

    return y_out


def integrate_adaptive(
        rhs, jac, dfunc_dy, ftimes, y0, tol=1e-6
):
    if isinstance(y0, numbers.Number):
        y0 = np.array([y0], dtype=np.float64)
    elif not isinstance(y0, np.ndarray):
        raise TypeError('y0 should be a number or ndarray')

    total_error = 2*tol
    times = np.linspace(ftimes[0], ftimes[-1], 2)

    while total_error > tol:
        T = len(times)

        print('integrate using times')
        print(times)
        y = integrate(rhs, times, y0, method=runge_kutta4)

        total_error, error = adjoint_error(
            rhs, jac, dfunc_dy, ftimes, times, y
        )
        print('error {}'.format(total_error))
        print(error)

        # if the error is above tolerance:
        # - split those segments that have an error > (tol / T)
        # - (TODO) join segments whose combined error < alpha * (tol / T)
        # alpha = 0.1
        if total_error > tol:
            print('splitting')
            split = error > tol / T
            print(split)
            T += np.sum(split)
            new_times = np.empty(T, dtype=times.dtype)
            index = 0
            for i in range(len(split)):
                new_times[index] = times[i]
                index += 1
                if split[i]:
                    new_times[index] = 0.5*(times[i] + times[i+1])
                    index += 1
            new_times[-1] = times[-1]
            times = new_times

            print('new times')
            print(times)

    return y, times

def adjoint_error(
        rhs, jac, dfunc_dy, ftimes, times, y, method=runge_kutta5
):

    T = len(times)
    fT = len(ftimes)
    n = y.shape[1]

    # calculate the derivative of the function wrt y
    Ju = dfunc_dy(interpolate(y, times, rhs, ftimes))
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
        error[i] = phi_error[-1] - phi_error1

    return phi_error[-1], error


def adjoint_sensitivities(
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

    def adjoint_sensitivities_error(t, phi_error, y_interp):
        phi = phi_error[:n]
        return np.concatenate((
            -jac_s(t, y_interp(t), phi),
            (rhs_s(t, y_interp(t)) - y_interp.grad(t)) * phi,
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


