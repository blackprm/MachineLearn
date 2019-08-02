from playML.LinearRegression import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()
x = boston.data
y = boston.target
x = x[y < 50.0]
y = y[y < 50.0]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2,random_state=666)
classfi = LinearRegression()

classfi.fit_normal(X_train, Y_train)
predict = classfi.predict(X_test)

socre = classfi.socre(Y_test, predict)

