import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(666)
x = 2 * np.random.random(size=100)
y = x * 3. + 4. + np.random.normal(size=100)
x = x.reshape(-1, 1)
ss = np.ones([len(x), 1])
X_b = np.hstack((ss, x))

'''
正规方程法进行回归
x = x.reshape(-1, 1)
regression = LinearRegression()
regression.fit(x, y)
predict = regression.predict(x[2:, :])
'''

'''
    梯度下降法进行训练
'''


def J(theta, X_b, Y):
    """
    求损失函数的函数值 -> funcValue
    :param theta: 预测的参数
    :param X_b: 数据的特征
    :param Y: 数据的真实值
    :return: 损失函数的值
    """
    try:
        return np.sum((Y - X_b.dot(theta)) ** 2)
    except:
        return float('inf')


def dJ(theta, X_b, Y):
    """
    求损失函数的导数值 -> d
    :param theta: 拟合参数
    :param X_b: 数据特征
    :param Y: 数据结果
    :return:     求损失函数的导数值 -> d
    """
    res = np.empty(len(theta))
    res[0] = np.sum(X_b.dot(theta) - Y)
    for i in range(1, len(theta)):
        res[i] = (X_b.dot(theta) - Y).dot(X_b[:, i])
    return res * 2 / len(theta)


def gradient_descent(X_b, Y, inital_theta, eta, n_iters=1e4, epsilon=1e-8):
    """
    梯度下降法拟合参数
    :param X_b:
    :param inital_theta:  初始参数
    :param eta:     学习率
    :param n_iters: 迭代次数
    :param epsilon: 精度
    :return:
    """

    theat = inital_theta
    i_iter = 0
    while i_iter < n_iters:
        gradient = dJ(theat, X_b, Y)
        last_theta = theat
        print(gradient)
        theat = theat - eta * gradient
        if abs(J(theat, X_b, Y) - J(last_theta, X_b, Y)) < epsilon:
            break
        i_iter += 1
    return theat


inital_theta = np.zeros(X_b.shape[1]) # 初始化拟合参数
eta = 0.0001

descent = gradient_descent(X_b, y, inital_theta, eta) # 梯度下降后的参数值

plt.scatter(x, y)
array = np.array([[0], [5]])
predict = array * descent[1] + descent[0]
plt.plot(array, predict, color='r')
plt.show()
