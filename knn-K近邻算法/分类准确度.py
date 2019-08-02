import matplotlib

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
test_ration = 0.2

digits = ds.load_digits()
X = digits.data
Y = digits.target

data_len = len(X) # 数据集的总长度
test_data_size = int(data_len * test_ration)
shuffle_index = np.random.permutation(data_len)

train_index = shuffle_index[test_data_size:]
test_index = shuffle_index[:test_data_size]

x_train = X[train_index]
y_train = Y[train_index]

x_test = X[test_index]
y_test = Y[test_index]

classifier = KNeighborsClassifier(n_neighbors=6)
classifier.fit(x_train,y_train)
predict = classifier.predict(x_test)

true_predicr = sum(predict == y_test)
print(true_predicr / y_test.size)
# some_digits = x.reshape(8, 8)
# plt.imshow(some_digits,matplotlib.cm.binary)
# plt.show()

