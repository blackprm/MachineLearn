import numpy as np
from sklearn.metrics import r2_score

class LinearRegression:
    '''
        多元线性回归
    '''
    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def fit_normal(self,x_train, y_train):
        X_b = np.hstack([np.ones((len(x_train), 1)),x_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, X):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        return X_b.dot(self._theta)

    def socre(self, y_true, y_predict):
        return r2_score(y_true, y_predict)

    def __repr__(self):
        return "LinearRegression()"
