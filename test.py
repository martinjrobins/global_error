from integrate import (
    integrate,
    interpolate,
    integrate_adaptive,
    adjoint_sensitivities_single_times,
    adjoint_error_single_times,
    adjoint_error, adjoint_error_and_sensitivities,
    adjoint_error_and_sensitivities_interp,
    adjoint_error_and_sensitivities_independent_ftimes,
    Minimise, MinimiseTraditional, MinimiseTraditionalNoGradient,
    MinimiseNonFixed, MinimiseMaxAdapt, MinimiseInterp
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
import scipy.integrate
import time

class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args):
        self.count += 1
        return self.func(*args)



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

    @unittest.skip("not working yet")
    def test_integrate_logistic_adaptive(self):
        r = 1.0
        k = 1.0

        def rhs(t, u):
            return r * u * (1 - u / k)

        def jac(t, u, x):
            return r * (1 - 2 * u / k) * x

        u0 = 0.1
        t = np.linspace(0, 1, 100)
        y = integrate(rhs, t, u0)
        y_exp = y + np.random.normal(scale=0.05, size=y.shape)

        def functional(y):
            return np.sum((y - y_exp)**2)

        def dfunc_dy(y):
            return 2 * (y - y_exp)

        y = integrate_adaptive(rhs, jac, dfunc_dy, t, u0, tol=1e-6)
        analytic = k / (1 + (k / u0 - 1) * np.exp(-r * t))\
            .reshape(-1, 1)
        np.testing.assert_allclose(
            functional(analytic), functional(y), rtol=1e-6, atol=0
        )

    def test_logistic_adjoint_sensitivities(self):
        r = 1.0
        k = 1.0
        def analytic(t, p, u0):
            return (
                p[1] / (1 + (p[1]/ u0 - 1) * np.exp(-p[0] * t))
            ).reshape(-1, 1)

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
        analytic_t = analytic(t, (r, k), u0)
        np.testing.assert_allclose(
            analytic_t, y, rtol=1e-5, atol=0
        )
        np.random.seed(0)
        y_exp = y + np.random.normal(scale=0.05, size=y.shape)

        def functional(y):
            return np.sum((y - y_exp)**2)

        def dfunc_dy(y):
            return 2 * (y - y_exp)

        dfdp = adjoint_sensitivities_single_times(
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

        t = np.linspace(0, 12.0, 100)
        y = integrate(rhs, t, u0)

        ft = np.linspace(0, 12.0, 10)
        analytic_exp = analytic(ft, (r, k), u0)
        y_exp = analytic_exp + \
            np.random.normal(scale=0.15, size=analytic_exp.shape)


        fy = interpolate(y, t, rhs, ft)
        _, _, f, g = adjoint_error_and_sensitivities_independent_ftimes(
            rhs, jac, drhs_dp, functional, dfunc_dy, fy, ft, t, y
        )

        analytic_dydr = k * ft * (k / u0 - 1) * np.exp(-r * ft) / \
            ((k / u0 - 1) * np.exp(-r * ft) + 1)**2
        analytic_dydk = -k * np.exp(-r * ft) / \
            (u0 * ((k / u0 - 1) * np.exp(-r * ft) + 1)**2) \
            + 1 / ((k / u0 - 1) * np.exp(-r * ft) + 1)
        analytic_dydp = np.stack(
            (analytic_dydr, analytic_dydk), axis=1
        )
        np.testing.assert_allclose(
            np.sum(dfunc_dy(analytic_exp)*analytic_dydp, axis=0),
            g, rtol=1e-4, atol=0
        )
        np.testing.assert_allclose(
            functional(analytic_exp),
            f, rtol=1e-6, atol=0
        )

        y_exp_interp = scipy.interpolate.interp1d(ft, y_exp, axis=0, kind='cubic')

        def functional_interp(t, y):
            return np.sum((y - y_exp_interp(t))**2)

        def dfunc_dy_interp(t, y):
            return 2 * (y - y_exp_interp(t))

        _, _, f, g = adjoint_error_and_sensitivities_interp(
            rhs, jac, drhs_dp, functional_interp, dfunc_dy_interp, t, y
        )

        def analytic_f_func(t):
            analytic_exp = analytic(t, (r, k), u0)
            return functional_interp(t, analytic_exp)
        def analytic_gr_func(t):
            analytic_exp = analytic(t, (r, k), u0)
            analytic_dydr = k * t * (k / u0 - 1) * np.exp(-r * t) / \
                ((k / u0 - 1) * np.exp(-r * t) + 1)**2
            return dfunc_dy_interp(t, analytic_exp)*analytic_dydr
        def analytic_gk_func(t):
            analytic_exp = analytic(t, (r, k), u0)
            analytic_dydk = -k * np.exp(-r * t) / \
                (u0 * ((k / u0 - 1) * np.exp(-r * t) + 1)**2) \
                + 1 / ((k / u0 - 1) * np.exp(-r * t) + 1)
            return dfunc_dy_interp(t, analytic_exp)*analytic_dydk


        analytic_gr = scipy.integrate.quad(analytic_gr_func, 0, 12)[0]
        analytic_gk = scipy.integrate.quad(analytic_gk_func, 0, 12)[0]
        analytic_f = scipy.integrate.quad(analytic_f_func, 0, 12)[0]

        np.testing.assert_allclose(
            [analytic_gr, analytic_gk], g, rtol=1e-4, atol=0
        )
        np.testing.assert_allclose(
            analytic_f,
            f, rtol=1e-6, atol=0
        )

    def test_logistic_adjoint_error(self):
        def analytic(t, p, u0):
            return (
                p[1] / (1 + (p[1]/ u0 - 1) * np.exp(-p[0] * t))
            ).reshape(-1, 1)

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
        analytic_t = analytic(t, (r, k), u0)
        np.testing.assert_allclose(
            analytic_t, y, rtol=1e-3, atol=0
        )
        np.random.seed(0)
        y_exp = y + np.random.normal(scale=0.05, size=y.shape)

        def functional(y):
            return np.sum((y - y_exp)**2)

        def dfunc_dy(y):
            return 2 * (y - y_exp)

        error, _ = adjoint_error_single_times(
            rhs, jac, dfunc_dy, t, y
        )

        np.testing.assert_allclose(
            functional(y) - functional(analytic_t),
            error, rtol=6e-3, atol=0
        )

        t = np.linspace(0, 12.0, 20)
        y = integrate(rhs, t, u0)
        ft = np.linspace(0, 12.0, 15)
        analytic_exp = analytic(ft, (r, k), u0)
        y_exp = analytic_exp + \
            np.random.normal(scale=0.15, size=analytic_exp.shape)

        fy = interpolate(y, t, rhs, ft)
        total_error, error, f, _ = \
            adjoint_error_and_sensitivities_independent_ftimes(
                rhs, jac, drhs_dp, functional, dfunc_dy, fy, ft, t, y
            )
        print(f)

        np.testing.assert_allclose(
            total_error, np.sum(error),
            rtol=1e-6
        )

        np.testing.assert_allclose(
            f - functional(analytic_exp),
            total_error, rtol=3e-2, atol=0
        )

        y_exp_interp = scipy.interpolate.interp1d(ft, y_exp, axis=0, kind='cubic')

        def functional_interp(t, u):
            return np.sum((u - y_exp_interp(t))**2)

        def dfunc_dy_interp(t, u):
            return 2 * (u - y_exp_interp(t))

        total_error, error, f, _ = adjoint_error_and_sensitivities_interp(
            rhs, jac, drhs_dp, functional_interp, dfunc_dy_interp, t, y
        )
        print('f = ', f)

        np.testing.assert_allclose(
            total_error, np.sum(error),
            rtol=1e-6
        )

        analytic_f, analytic_err = scipy.integrate.quad(
            lambda it: functional_interp(it, analytic(it, (r, k), u0)), 0, 12.0
        )

        print('analytic f = ', analytic_f)
        print('error = ', error)
        print('total error = ', total_error)

        plt.plot(t, y, label='sim')
        ft = np.linspace(0, 12, 1000)
        plt.plot(ft, y_exp_interp(ft), label='exp')
        plt.show()

        np.testing.assert_allclose(
            f - analytic_f,
            total_error, rtol=1e-2, atol=0
        )

        errors = []
        ns = [10, 100, 1000, 10000]
        for n in ns:
            t = np.linspace(0, 12.0, n)
            y = integrate(rhs, t, u0)
            ft = np.linspace(0, 12.0, 13)
            analytic_exp = analytic(ft, (r, k), u0)
            y_exp = analytic_exp + \
                np.random.normal(scale=0.15, size=analytic_exp.shape)

            fy = interpolate(y, t, rhs, ft)
            total_error, error, f, _ = \
                adjoint_error_and_sensitivities_independent_ftimes(
                    rhs, jac, drhs_dp, functional, dfunc_dy, fy, ft, t, y
                )
            errors.append(total_error/f)

        print(errors)
        plt.loglog(ns, np.abs(errors))
        plt.loglog(ns, np.array(ns, dtype=float)**(-4), label='n^-4')
        plt.legend()
        plt.savefig('rel_error_versus_n.pdf')


    def test_interpolate(self):
        r = 1.0
        k = 1.0

        def rhs(t, u):
            return r * u * (1 - u / k)

        u0 = 0.1
        t = np.linspace(0, 1, 100)
        y = integrate(rhs, t, u0)

        interp_t = np.linspace(0, 1, 33)
        interp_y = interpolate(y, t, rhs, interp_t)

        analytic = k / (1 + (k / u0 - 1) * np.exp(-r * interp_t))\
            .reshape(-1, 1)

        np.testing.assert_allclose(
            analytic, interp_y, rtol=1e-5, atol=0
        )

        interp_t = np.linspace(0, 1, 133)
        interp_y = interpolate(y, t, rhs, interp_t)

        analytic = k / (1 + (k / u0 - 1) * np.exp(-r * interp_t))\
            .reshape(-1, 1)

        np.testing.assert_allclose(
            analytic, interp_y, rtol=1e-5, atol=0
        )


    def test_adjoint_error(self):
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

        np.random.seed(0)
        for ntimes in [13, 23]:
            ft = np.linspace(0, 10.0, ntimes)
            fy = interpolate(y, t, rhs, ft)
            analytic = k / (1 + (k / u0 - 1) * np.exp(-r * ft))\
                .reshape(-1, 1)
            y_exp = fy + np.random.normal(scale=0.05, size=fy.shape)

            def functional(y):
                return np.sum((y - y_exp)**2)

            def dfunc_dy(y):
                return 2 * (y - y_exp)

            error, errors, f, _, _ = adjoint_error(
                rhs, jac, functional, dfunc_dy, ft, t, y
            )

            np.testing.assert_allclose(
                f - functional(analytic),
                error, rtol=7e-2, atol=1e-8
            )

    def test_adaptive_integrate(self):
        r = 1.0
        k = 1.0

        def rhs(t, u):
            return r * u * (1 - u / k)

        def jac(t, u, x):
            return r * (1 - 2 * u / k) * x

        u0 = 0.1
        np.random.seed(0)
        for fn in [2, 5, 7, 13, 133]:
            ft = np.linspace(0, 10.0, fn)
            k_exp = k * 0.9
            r_exp = r * 0.9
            analytic_exp = k_exp / (1 + (k_exp / u0 - 1) * np.exp(-r_exp * ft))\
                .reshape(-1, 1)
            y_exp = analytic_exp + np.random.normal(scale=0.05, size=analytic_exp.shape)
            fanalytic = k / (1 + (k / u0 - 1) * np.exp(-r * ft))\
                .reshape(-1, 1)

            def functional(y):
                return np.sum((y - y_exp)**2)

            def dfunc_dy(y):
                return 2 * (y - y_exp)


            y, t, f, rhs_eval, jac_eval = integrate_adaptive(
                rhs, jac, functional, dfunc_dy, ft, u0, rtol=1e-4
            )
            print('rhs_eval = {}, jac_eval = {}'.format(
                rhs_eval, jac_eval))

            sol = scipy.integrate.solve_ivp(
                rhs, (ft[0], ft[-1]), y0=[u0], rtol=1e-6
            )
            fsol = functional(interpolate(sol.y.T, sol.t, rhs, ft))
            print('scipy: rhs_eval = {}, jac_eval = {}'.format(
                sol.nfev, sol.njev))



            print(len(t), len(sol.t))
            print('scipy error',
                 (fsol - functional(fanalytic)) / functional(fanalytic)
            )
            print('my error',
                 (f - functional(fanalytic)) / functional(fanalytic)
            )
            #plt.plot(t, y, '.', label='mine')
            #plt.plot(sol.t, sol.y.reshape(-1), '.', label='scipy')
            #plt.legend()
            #plt.show()

            np.testing.assert_allclose(
                f, functional(fanalytic), rtol=2e-4, atol=0
            )



    def test_minimise(self):
        def analytic(t, p, u0):
            return (
                p[1] / (1 + (p[1]/ u0 - 1) * np.exp(-p[0] * t))
            ).reshape(-1, 1)

        def rhs(t, u, p):
            return p[0] * u * (1 - u / p[1])

        def jac(t, u, x, p):
            return p[0] * (1 - 2 * u / p[1]) * x

        def drhs_dp(t, u, x, p):
            return np.array([
                u * (1 - u / p[1]),
                p[0] * u**2 / p[1]**2,
            ]).dot(x)

        u0 = 0.01
        np.random.seed(5)
        fn = 16
        ft = np.linspace(0, 12.0, fn)
        k_exp = 0.9
        r_exp = 1.9
        p0 = [0.5, 1.5]
        bounds = [(0.01, 3), (0.01, 3)]
        analytic_exp = analytic(ft, (r_exp, k_exp), u0)
        y_exp = analytic_exp + np.random.normal(scale=0.05, size=analytic_exp.shape)
        y_exp_interp = scipy.interpolate.interp1d(ft, y_exp, axis=0, kind='cubic')

        def functional(y):
            return np.sum((y - y_exp)**2)

        def dfunc_dy(y):
            return 2 * (y - y_exp)

        def functional_interp(t, y):
            return np.sum((y - y_exp_interp(t))**2)

        def dfunc_dy_interp(t, y):
            return 2 * (y - y_exp_interp(t))


        # MinimiseMaxAdapt
        print('running MinimiseMaxAdapt')
        rhs_tracked = CountCalls(rhs)
        jac_tracked = CountCalls(jac)
        minimise_adapt = MinimiseMaxAdapt(
            rhs_tracked, jac_tracked,
            drhs_dp, functional, dfunc_dy, ft, u0,
            rtol=1e-4, atol=1e-4
        )

        t0 = time.perf_counter()
        res = scipy.optimize.minimize(
            minimise_adapt, p0, jac=True, bounds=bounds
        )
        t1 = time.perf_counter()
        print(res.status)
        print(res.message)
        print('final p = {}, #rhs = {}, #jac = {}, time = {}'.format(
            res.x, rhs_tracked.count, jac_tracked.count, t1-t0
        ))
        analytic_fit = analytic(minimise_adapt.times, res.x, u0)

        plt.clf()
        plt.plot(minimise_adapt.times, analytic_fit, '.-', label='fit')
        plt.plot(ft, y_exp, '.', label='data')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.title(
            'adaptive minimiser (#rhs = {}, #jac = {})'.format(
                rhs_tracked.count, jac_tracked.count
            )
        )
        plt.legend()
        plt.savefig('test_minimise_adaptive_fit.pdf')

        # MinimiseNonFixed
        print('running MinimiseNonFixed')
        rhs_tracked = CountCalls(rhs)
        jac_tracked = CountCalls(jac)
        minimise_nonfixed = MinimiseNonFixed(
            rhs_tracked, jac_tracked,
            drhs_dp, functional, dfunc_dy, ft, u0,
            rtol=1e-4, atol=1e-4
        )

        t0 = time.perf_counter()
        res = scipy.optimize.minimize(
            minimise_nonfixed, p0, jac=True, bounds=bounds
        )
        t1 = time.perf_counter()
        print(res.status)
        print(res.message)
        print('final p = {}, #rhs = {}, #jac = {}, time = {}'.format(
            res.x, rhs_tracked.count, jac_tracked.count, t1-t0
        ))
        analytic_fit = analytic(minimise_nonfixed.times, res.x, u0)

        plt.clf()
        plt.plot(minimise_nonfixed.times, analytic_fit, '.-', label='fit')
        plt.plot(ft, y_exp, '.', label='data')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.title(
            'adaptive nonfixed minimiser (#rhs = {}, #jac = {})'.format(
                rhs_tracked.count, jac_tracked.count
            )
        )
        plt.legend()
        plt.savefig('test_minimise_adaptive_nonfixed_fit.pdf')


        # MinimiseInterp
        print('running MinimiseInterp')
        rhs_tracked = CountCalls(rhs)
        jac_tracked = CountCalls(jac)
        minimise_interp = MinimiseInterp(
            rhs_tracked, jac_tracked,
            drhs_dp, functional_interp, dfunc_dy_interp, ft, u0,
            rtol=1e-4, atol=1e-4
        )

        t0 = time.perf_counter()
        res = scipy.optimize.minimize(
            minimise_interp, p0, jac=True, bounds=bounds
        )
        t1 = time.perf_counter()
        print(res.status)
        print(res.message)
        print('final p = {}, #rhs = {}, #jac = {}, time = {}'.format(
            res.x, rhs_tracked.count, jac_tracked.count, t1-t0
        ))
        analytic_fit = analytic(minimise_interp.times, res.x, u0)

        plt.clf()
        plt.plot(minimise_interp.times, analytic_fit, '.-', label='fit')
        plt.plot(ft, y_exp, '.', label='data')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.title(
            'adaptive minimiser interp (#rhs = {}, #jac = {})'.format(
                rhs_tracked.count, jac_tracked.count
            )
        )
        plt.legend()
        plt.savefig('test_minimise_adaptive_interp_fit.pdf')

        print('running MinimiseTraditional')
        rhs_tracked = CountCalls(rhs)
        jac_tracked = CountCalls(jac)
        minimise_trad = MinimiseTraditional(
            rhs_tracked, jac_tracked,
            drhs_dp, functional, dfunc_dy, ft, [u0],
            rtol=1e-5, atol=1e-5
        )

        t0 = time.perf_counter()
        res = scipy.optimize.minimize(
            minimise_trad, p0, jac=True, bounds=bounds
        )
        t1 = time.perf_counter()
        print(res.status)
        print(res.message)
        print('final p = {}, #rhs = {}, #jac = {}, time = {}'.format(
            res.x, rhs_tracked.count, jac_tracked.count, t1-t0
        ))
        analytic_fit = analytic(minimise_trad.times, res.x, u0)

        plt.clf()
        plt.plot(minimise_trad.times, analytic_fit, '.-', label='fit')
        plt.plot(ft, y_exp, '.', label='data')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.title(
            'traditional minimiser (#rhs = {}, #jac = {})'.format(
                rhs_tracked.count, jac_tracked.count
            )
        )
        plt.legend()
        plt.savefig('test_minimise_traditional_fit.pdf')


        plt.clf()
        plt.subplot(1, 2, 1)
        plt.semilogy(
            np.abs(np.array(minimise_adapt.total_error) / minimise_adapt.f_evals),
            '-', label='adaptive')
        plt.semilogy(
            np.abs(np.array(minimise_interp.total_error) / minimise_interp.f_evals),
            '-', label='adaptive interp')
        plt.semilogy(
            np.abs(np.array(minimise_trad.total_error) / minimise_trad.f_evals),
            '-', label='traditional')
        plt.semilogy(
            np.abs(np.array(minimise_nonfixed.total_error) / minimise_nonfixed.f_evals),
            '-', label='adaptive nonfixed')

        plt.xlabel('function eval #')
        plt.ylabel('total error using adjoint')
        plt.subplot(1, 2, 2)
        plt.plot(minimise_adapt.ntimes, '-',
                 label='adapt')
        plt.plot(minimise_interp.ntimes, '-',
                 label='adapt interp')
        plt.plot(minimise_trad.ntimes, '-',
                 label='traditional')
        plt.plot(minimise_nonfixed.ntimes, '-',
                 label='adapt nonfixed')
        plt.xlabel('function eval #')
        plt.ylabel('# time points for ode solve')
        plt.legend()
        plt.savefig('test_minimise_comparison.pdf')

        print('running MinimiseTraditionalNoGradient')
        rhs_tracked = CountCalls(rhs)
        jac_tracked = CountCalls(jac)
        minimise_trad_no_grad = MinimiseTraditionalNoGradient(
            rhs_tracked, jac_tracked,
            drhs_dp, functional, dfunc_dy, ft, [u0],
            rtol=1e-4, atol=1e-4
        )
        t0 = time.perf_counter()
        res = scipy.optimize.minimize(
            minimise_trad_no_grad, p0, bounds=bounds
        )
        t1 = time.perf_counter()
        print(res.status)
        print(res.message)
        print('final p = {}, #rhs = {}, #jac = {}, time = {}'.format(
            res.x, rhs_tracked.count, jac_tracked.count, t1-t0
        ))



if __name__ == '__main__':
    unittest.main()
