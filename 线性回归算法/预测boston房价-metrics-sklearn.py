import numpy as np

import sklearn.datasets as ds
import matplotlib.pyplot as plt
import playML.SimpleLinearRegression1 as SLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mean_absolute_error_, mean_squared_error as mean_squared_error_
from sklearn.metrics import r2_score

boston = ds.load_boston()
x = boston.data[:, 5]
y = boston.target

x = x[y < 50.0]
y = y[y < 50.0]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=666)

regression_ = SLR.SimpleLinearRegression2()
regression_.fit(X_train, Y_train)

predict = regression_.predict(X_test)

plt.scatter(x, y, color='pink')
plt.scatter(X_test, predict, color='red')
plt.scatter(X_test, Y_test, color='green')
plt.show()

'''
    MSE
'''
mean_squared_error = ((Y_test - predict) ** 2).sum() / len(Y_test)

sk_squared_error_ = mean_squared_error_(Y_test, predict)
'''
    RMSE
'''
root_mean_squared_error = np.sqrt(mean_squared_error)

'''
MAE
'''
mean_absolute_error = np.abs(Y_test - predict).sum() / len(Y_test)
sk_absolute_error_ = mean_absolute_error_(Y_test, predict)

'''
R2
'''

R2 = 1 - mean_squared_error_(Y_test, predict) / np.var(Y_test)
sk_r2 = r2_score(Y_test, predict)
