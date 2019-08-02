import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
iris = ds.load_iris()
X = iris.data
Y = iris.target

data_len = len(X)
shuffle_index = np.random.permutation(data_len) # 求随机排列
test_ration = 0.2
test_size = int(data_len * test_ration)
test_indexes = shuffle_index[:test_size]
train_indexes = shuffle_index[test_size:]

X_train = X[train_indexes]
Y_train = Y[train_indexes]

X_test = X[test_indexes]
Y_test = Y[test_indexes]

classifier = KNeighborsClassifier(n_neighbors=6)
classifier.fit(X_train, Y_train)
predict = classifier.predict(X_test)
ret = (predict == Y_test)
counter = Counter(ret)

for key in counter:
    if key:
        print("正确率: " + str(counter[key] / test_size))
        