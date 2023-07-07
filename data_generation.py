import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def lotka_volterra(state, t):
    x, y = state
    alpha, beta, gamma, delta = 0.1, 0.1, 0.1, 0.1
    dx = alpha * x - beta * x * y
    dy = delta * x * y - gamma * y
    return dx, dy

t = np.linspace(0, 50, 500)
init_state = [10, 5]
sol = odeint(lotka_volterra, init_state, t)

plt.plot(t, sol[:, 0], label="Prey")
plt.plot(t, sol[:, 1], label="Predator")
plt.legend()
plt.show()
