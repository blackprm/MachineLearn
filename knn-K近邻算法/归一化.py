import numpy as np
from matplotlib import pyplot

vector = np.random.randint(0, 100, size=100)
_min = min(vector)
_max = max(vector)

# 均值归一化
normal_vector = (vector - _min) * 1.0 / (_max - _min)
print(normal_vector)

avg = np.mean(vector) # 均值
std = np.std(vector) # 方差
# 方差归一化
std_vector = (vector - avg)  / std
pyplot.scatter(std_vector, std_vector)
pyplot.show()