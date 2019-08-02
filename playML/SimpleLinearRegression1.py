import numpy as np
from sklearn.metrics import mean_squared_error


class SimpleLinearRegression1:

    def __init__(self):
        """
        构造函数
        """
        self.a_ = None
        self.b_  = None


    def fit(self, x_train, y_train):
        '''
        根据训练集训练模型
        :param x_train:
        :param y_train:
        :return:
        '''

        assert x_train.ndim == 1, \
        "Simple Linear Regression can only solve single featute training data."
        assert len(x_train) == len(y_train),\
        "the size of x_train must be equals with the size of y_train."
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0

        # a = ∑ (Xi - X_mean)(Yi - Y_mean) / ∑ (xi - X_mean)2

        for x_i, y_i in zip(x_train, y_train):
            num += ((x_i - x_mean) * (y_i - y_mean))
            d += (x_i - x_mean) ** 2
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self,x_predict):
        '''
        预测向量
        :param x_predict:
        :return:
        '''

        return self.a_ *x_predict + self.b_



class SimpleLinearRegression2:

    def __init__(self):
        """
        构造函数
        """
        self.a_ = None
        self.b_  = None


    def fit(self, x_train, y_train):
        '''
        根据训练集训练模型
        :param x_train:
        :param y_train:
        :return:
        '''

        assert x_train.ndim == 1, \
        "Simple Linear Regression can only solve single featute training data."
        assert len(x_train) == len(y_train),\
        "the size of x_train must be equals with the size of y_train."

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0

        # a = ∑ (Xi - X_mean)(Yi - Y_mean) / ∑ (xi - X_mean)2

        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self,x_predict):
        '''
        预测向量
        :param x_predict:
        :return:
        '''

        return self.a_ *x_predict + self.b_


    def r2_socre(self, y_true, y_predict):
        return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)
