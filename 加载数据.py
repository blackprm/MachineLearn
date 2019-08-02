import sklearn.datasets as sk
import matplotlib.pyplot as plt
import cv2

iris = sk.load_iris()
print(iris.keys())
datas = iris.data[:, 2:4]
y = iris.target
plt.scatter(datas[y == 0, 0], datas[y == 0, 1], color='red', marker="o")
plt.scatter(datas[y == 1, 0], datas[y == 1, 1], color='green', marker="x")
plt.scatter(datas[y == 2, 0], datas[y == 2, 1], color='blue', marker="+")
plt.show()
