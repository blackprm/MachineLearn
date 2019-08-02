# knn算法实例
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
raw_data_X = [
    [3.3, 2.3],
    [3.1, 1.7],
    [1.3, 3.3],
    [3.5, 4.6],
    [2.2, 2.8],
    [7.4, 4.6],
    [5.7, 3.5],
    [9.1, 2.5],
    [7.7, 3.4],
    [7.9, 0.79]
]

raw_data_Y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

train_x = np.array(raw_data_X)
train_y = np.array(raw_data_Y)

KNN_Classifier = KNeighborsClassifier(n_neighbors=6)


# 集合数据
KNN_Classifier.fit(train_x, train_y)
x = np.array([8.0, 3.3])
X_predict = x.reshape(1, -1)
# 进行预测数据
predict = KNN_Classifier.predict(X_predict)
print(predict[0])