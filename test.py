from runge_kutta import integrate
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


def rhs(t, y):
    return -y


times = np.linspace(0.0, 1.0, 100)
y, k = integrate(rhs, times, 1.0)
analytic = np.exp(-times)
error = np.sum((analytic - y)**2) / np.sum(analytic**2)
print('error', error)
plt.plot(times, y, label='sim')
plt.plot(times, analytic, label='analytic')
plt.legend()
plt.show()

