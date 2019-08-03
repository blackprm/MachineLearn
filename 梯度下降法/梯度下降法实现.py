import numpy as np
import matplotlib.pyplot as plt


def dJ(theta):
    return 2 * (theta - 2.5)


def J(theta):
    return (theta - 2.5) ** 2 - 1


plot_x = np.linspace(-1, 6, 141)
plot_y = (plot_x - 2.5) ** 2 - 1

theta = 0
eta = 0.1
theta_history = [theta]
while True:
    gradient = dJ(theta)
    theta = theta - eta * gradient
    theta_history.append(theta)
    if abs(gradient) < 1e-8:
        break

plt.plot(plot_x, plot_y)
plt.scatter(np.array(theta_history), J(np.array(theta_history)), color='r', marker='+')
plt.show()
