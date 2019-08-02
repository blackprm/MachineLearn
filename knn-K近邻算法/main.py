import math
'''
    knn
    fit 数据
    predict 预测数据
    不需要训练数据,没有模型
    缺点: 每次都需要巨大的计算量
'''

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
k = 6
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

np_x = np.array(raw_data_X)
np_y = np.array(raw_data_Y)

plt.scatter(np_x[np_y == 0, 0], np_x[np_y == 0, 1], color='red')
plt.scatter(np_x[np_y == 1, 0], np_x[np_y == 1, 1], color='green')
x = np.array([8.0, 3.3])
plt.scatter(x[0], x[1], color='blue')
plt.show()

distance = [math.sqrt(np.sum(i - x)**2)for i in np_x]
nearest = np.argsort(distance)
topK_y = [np_y[i] for i in nearest[:k]]
counter = Counter(topK_y)
predict_y = counter.most_common(1)[0][0]
print(predict_y)