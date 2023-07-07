import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Define time discretization
t_min, t_max = 0, 50
t_count = 1000
t = np.linspace(t_min, t_max, t_count)
dt = (t_max - t_min) / (t_count - 1)

# Define parameters for the Lotka-Volterra equations
alpha, beta, delta, gamma = 2/3, 4/3, 1, 1

# Define initial condition
x0, y0 = 1, 1

# Generate training data (i.e., the exact solution of the ODE)
Y = np.zeros((t_count, 2))
Y[0, 0], Y[0, 1] = x0, y0
for i in range(1, t_count):
    Y[i, 0] = Y[i - 1, 0] + dt * (alpha * Y[i - 1, 0] - beta * Y[i - 1, 0] * Y[i - 1, 1])
    Y[i, 1] = Y[i - 1, 1] + dt * (delta * Y[i - 1, 0] * Y[i - 1, 1] - gamma * Y[i - 1, 1])

# Normalize the data
Y_norm = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)

# Define the LSTM model
model = Sequential([
    LSTM(20, return_sequences=True, input_shape=[None, 2]),
    LSTM(20),
    Dense(2)
])

# Compile and train the model
model.compile(loss="mse", optimizer="adam")
history = model.fit(Y_norm[:-1].reshape(-1, 1, 2), Y_norm[1:].reshape(-1, 1, 2), epochs=200)

# Make predictions
Y_pred = model.predict(Y_norm[:-1].reshape(-1, 1, 2))

# Plot the results
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))

plt.plot(t, Y[:, 0], label="Prey (true)")
plt.plot(t, Y[:, 1], label="Predator (true)")
plt.plot(t[:-1], Y_pred[:, 0, 0]*np.std(Y, axis=0)[0] + np.mean(Y, axis=0)[0], "--", label="Prey (predicted)")
plt.plot(t[:-1], Y_pred[:, 0, 1]*np.std(Y, axis=0)[1] + np.mean(Y, axis=0)[1], "--", label="Predator (predicted)")

plt.legend()
plt.show()
