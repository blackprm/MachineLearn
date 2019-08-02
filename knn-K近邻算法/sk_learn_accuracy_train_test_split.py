from sklearn.model_selection import train_test_split
from sklearn import datasets as ds
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

digits = ds.load_digits()
X = digits.data
Y = digits.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=666)
classifier = KNeighborsClassifier(n_neighbors=6)
classifier.fit(X_train, Y_train)
predict = classifier.predict(X_test)

# 使用neighbors预测
classifier_score = classifier.score(X_test, Y_test)

# 使用sklearn accuracy_score 方法检测准确度
score = accuracy_score(Y_test, predict)

print(score)