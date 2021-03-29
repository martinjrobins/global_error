from integrate import (
    integrate,
    adjoint_sensitivities,
    adjoint_error,
)
from interpolate import CubicHermiteInterpolate
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import unittest
from runge_kutta import (
    runge_kutta5,
    runge_kutta4,
    runge_kutta41,
)


class TestGlobalError(unittest.TestCase):

    def test_integrate(self):
        def rhs(t, y):
            return -y

        times = np.linspace(0.0, 1.0, 100)
        for method in [runge_kutta4, runge_kutta41, runge_kutta5]:
            y = integrate(rhs, times, 1.0, method=method)
            analytic = np.exp(-times).reshape(-1, 1)
            np.testing.assert_allclose(
                analytic, y, rtol=1e-5, atol=0
            )

    def test_integrate2d(self):
        def rhs(t, y):
            return -y

        times = np.linspace(0.0, 1.0, 100)
        y0 = np.array([1.0, 2.0])
        for method in [runge_kutta4, runge_kutta41, runge_kutta5]:
            y = integrate(rhs, times, y0, method=method)

            analytic = np.stack(
                (np.exp(-times), 2*np.exp(-times)),
                axis=1
            )
            np.testing.assert_allclose(
                analytic, y, rtol=1e-5, atol=0
            )

    def test_interpolate(self):
        def rhs(t, y):
            return -y

        t0 = 0.1
        t1 = 0.2
        y0 = np.exp(-t0)
        y1 = np.exp(-t1)
        y = CubicHermiteInterpolate(
            t0, t1, y0, y1, rhs(t0, y0), rhs(t1, y1)
        )
        times = np.linspace(t0, t1, 100)
        interp_y = np.empty(len(times))
        for i in range(len(times)):
            interp_y[i] = y(times[i])
        analytic = np.exp(-times)
        np.testing.assert_allclose(analytic, interp_y, rtol=1e-5, atol=0)

    def test_interpolate_grad(self):
        def rhs(t, y):
            return -y

        t0 = 0.1
        t1 = 0.2
        y0 = np.exp(-t0)
        y1 = np.exp(-t1)
        y = CubicHermiteInterpolate(
            t0, t1, y0, y1, rhs(t0, y0), rhs(t1, y1)
        )
        times = np.linspace(t0, t1, 100)
        interp_grad = np.empty(len(times))
        for i in range(len(times)):
            interp_grad[i] = y.grad(times[i])
        analytic_grad = -np.exp(-times)
        np.testing.assert_allclose(
            analytic_grad, interp_grad, rtol=1e-5, atol=0)


    def test_interpolate2d(self):
        def rhs(t, y):
            return -y

        t0 = 0.1
        t1 = 0.2
        y0 = np.array([np.exp(-t0), 2*np.exp(-t0)])
        y1 = np.array([np.exp(-t1), 2*np.exp(-t1)])
        y = CubicHermiteInterpolate(
            t0, t1, y0, y1, rhs(t0, y0), rhs(t1, y1)
        )
        times = np.linspace(t0, t1, 100)
        interp_y = np.empty((len(times), len(y0)))
        for i in range(len(times)):
            interp_y[i] = y(times[i])
        analytic = np.stack(
            (np.exp(-times), 2*np.exp(-times)),
            axis=1
        )
        np.testing.assert_allclose(analytic, interp_y, rtol=1e-5, atol=0)

    def test_integrate_logistic(self):
        r = 1.0
        k = 1.0

        def rhs(t, u):
            return r * u * (1 - u / k)

        u0 = 0.1
        t = np.linspace(0, 1, 100)
        y = integrate(rhs, t, u0)
        analytic = k / (1 + (k / u0 - 1) * np.exp(-r * t))\
            .reshape(-1, 1)
        np.testing.assert_allclose(
            analytic, y, rtol=1e-5, atol=0
        )

    def test_logistic_adjoint_sensitivities(self):
        r = 1.0
        k = 1.0

        def rhs(t, u):
            return r * u * (1 - u / k)

        def jac(t, u, x):
            return r * (1 - 2 * u / k) * x

        def drhs_dp(t, u, x):
            return np.array([
                u * (1 - u / k),
                r * u**2 / k**2,
            ]).dot(x)

        u0 = 0.1
        t = np.linspace(0, 1.0, 1000)
        y = integrate(rhs, t, u0)
        analytic = k / (1 + (k / u0 - 1) * np.exp(-r * t))\
            .reshape(-1, 1)
        np.testing.assert_allclose(
            analytic, y, rtol=1e-5, atol=0
        )
        np.random.seed(0)
        y_exp = y + np.random.normal(scale=0.05, size=y.shape)

        def functional(y):
            return np.sum((y - y_exp)**2)

        def dfunc_dy(y):
            return 2 * (y - y_exp)

        dfdp = adjoint_sensitivities(
            rhs, jac, drhs_dp, dfunc_dy, t, y
        )

        analytic_dydr = k * t * (k / u0 - 1) * np.exp(-r * t) / \
            ((k / u0 - 1) * np.exp(-r * t) + 1)**2
        analytic_dydk = -k * np.exp(-r * t) / \
            (u0 * ((k / u0 - 1) * np.exp(-r * t) + 1)**2) \
            + 1 / ((k / u0 - 1) * np.exp(-r * t) + 1)
        analytic_dydp = np.stack(
            (analytic_dydr, analytic_dydk), axis=1
        )
        np.testing.assert_allclose(
            np.sum(dfunc_dy(y)*analytic_dydp, axis=0),
            dfdp, rtol=1e-6, atol=0
        )

    def test_logistic_adjoint_error(self):
        r = 1.0
        k = 1.0

        def rhs(t, u):
            return r * u * (1 - u / k)

        def jac(t, u, x):
            return r * (1 - 2 * u / k) * x

        def drhs_dp(t, u, x):
            return np.array([
                u * (1 - u / k),
                r * u**2 / k**2,
            ]).dot(x)

        u0 = 0.1
        t = np.linspace(0, 10.0, 20)
        y = integrate(rhs, t, u0)
        analytic = k / (1 + (k / u0 - 1) * np.exp(-r * t))\
            .reshape(-1, 1)
        np.testing.assert_allclose(
            analytic, y, rtol=1e-3, atol=0
        )
        np.random.seed(0)
        y_exp = y + np.random.normal(scale=0.05, size=y.shape)

        def functional(y):
            return np.sum((y - y_exp)**2)

        def dfunc_dy(y):
            return 2 * (y - y_exp)

        error = adjoint_error(
            rhs, jac, dfunc_dy, t, y
        )

        np.testing.assert_allclose(
            functional(y) - functional(analytic),
            error, rtol=2e-3, atol=0
        )


if __name__ == '__main__':
    unittest.main()
