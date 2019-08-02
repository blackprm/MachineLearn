import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = ds.load_iris()
X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
std = StandardScaler()
std.fit(X_train, Y_train)

mean = std.mean_ # 均值
std_ = std.scale_# 方差

X_train = std.transform(X_train)
X_test = std.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, Y_train)
predict = classifier.predict(X_test)

score = accuracy_score(Y_test, predict)
