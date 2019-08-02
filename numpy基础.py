import numpy as np

print(np.__version__)  # 版本
L = [i for i in range(10)]  # 对类型不做限定,但效率低
print(L)

import array

arr = array.array('i', [i for i in range(10)])  # array 中元素只能是一种，限定类型，提高效率
print(arr)

nparr = np.array([i for i in range(10)])  # 类型限定，提供向量方法
print(nparr)

# 生成随机矩阵
random_array = np.random.randint(0, 10, size=(3, 5))
print(random_array)

# 矩阵维度
v = np.random.normal(1, 1, 10)
print(v.ndim, v.shape, v.size)

# 子矩阵
M = np.random.normal(1, 20, size=(3, 5))
print(M[:2, :2])  # 采用的是指针复制的方法，所以效率高
print(M[:2, :2].copy())  # 采用深复制

# 矩阵的合并
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y])  # 向量拼接

A = np.zeros(shape=(2, 2))
A_A_Y = np.concatenate([A, A])  # 沿着行方向拼接
A_A_X = np.concatenate([A, A], axis=1)  # 沿着行方向拼接
print(A_A_Y)
print(A_A_X)

'''
数据的分割
'''
x = np.arange(25).reshape(
    5, 5
)
x1, x2, x3 = np.split(x, [1, 5])  # 按行分割
x1, x2, x3 = np.split(x, [1, 5], axis=1)  # 按列分割
print(x1, x2, x3)

'''
矩阵的运算
'''
a = np.array([0, 1, 2])
b = np.array([
    [1],
    [2],
    [3]
])

# 数乘/除
print(a * 2)

# 数学运算 ...
sinx = np.sin(a)
print(sinx)

# 矩阵乘法
c = a.dot(b)

# 矩阵转置
print(a.T)

# 矩阵的堆叠
v = [1, 2, 3]
tile = np.tile(v, (3, 1)) # 在行方向堆叠两次, 在列方向堆叠三次
print("tile = ", tile)

# 矩阵的逆运算
#inv_v = np.linalg.inv(tile)
#print(inv_v)


# 伪逆矩阵
prinV = np.linalg.pinv(x)
print(x.dot(prinV))


# 矩阵的rank
rank = np.linalg.matrix_rank(tile)
print(rank)

'''
    聚合操作
'''

rand = np.random.rand(100).reshape(4,-1)

# 求矩阵中所有元素的和
np_sum = np.sum(rand)
np_sum_Y = np.sum(rand, axis=1) # 按列求和

np_min = np.min(rand)
np_max = np.max(rand)
print(rand)
print(np_sum_Y)

'''
   arg运算 : 求运算的参数位置
'''

random_rand = np.random.random(100000)
print(random_rand.argmin())


# 排序和索引
x = np.arange(0, 16)
np.random.shuffle(x)
sort = np.sort(x) # 排序
print(sort)

# fancy indexing
inx = [3, 5, 8]
print(x[inx])
inx = [True,True,False]
res = x < 2
print(x[res])