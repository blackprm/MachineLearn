import numpy as np
import matplotlib.pyplot as plt
x = np.array([
    1.,
    2.,
    3.,
    4.,
    5.
])
y = np.array([
    1.,
    3.,
    2.,
    3.,
    5.
])

plt.scatter(x, y)
plt.show()

x_mean = np.mean(x)
y_mean = np.mean(y)

num = 0.0
d = 0.0

#a = ∑ (Xi - X_mean)(Yi - Y_mean) / ∑ (xi - X_mean)2

plt.scatter(x, y)#b = y_mean - a*x_mean
for x_i , y_i in zip(x, y):
    num += ((x_i - x_mean) * (y_i - y_mean))
    d += (x_i - x_mean) ** 2
a = num / d
b = y_mean - a*x_mean

y_predict = a * x + b
plt.plot(x, y_predict)
plt.show()