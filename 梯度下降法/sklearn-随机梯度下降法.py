from sklearn import datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
boston = ds.load_boston()

X = boston.data
Y = boston.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_standard = scaler.transform(X_train)
X_test_standard = scaler.transform(X_test)

# 随机梯度下降法
sgd_regression = SGDRegressor()
sgd_regression.fit(X_train_standard, Y_train)
print(sgd_regression.score(X_test_standard, Y_test))
