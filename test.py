from integrate import integrate
from interpolate import CubicHermiteInterpolate
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import unittest

class TestGlobalError(unittest.TestCase):

    def test_integrate(self):
        def rhs(t, y):
            return -y

        times = np.linspace(0.0, 1.0, 100)
        y = integrate(rhs, times, 1.0)
        analytic = np.exp(-times)
        np.testing.assert_allclose(analytic, y, rtol=1e-5, atol=0)

    def test_integrate2d(self):
        def rhs(t, y):
            return -y

        times = np.linspace(0.0, 1.0, 100)
        y0 = np.array([1.0, 2.0])
        y = integrate(rhs, times, y0)
        analytic = np.exp(-times)
        np.testing.assert_allclose(analytic, y, rtol=1e-5, atol=0)


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


if __name__ == '__main__':
    unittest.main()
