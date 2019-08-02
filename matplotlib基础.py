import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 3.14, 10000)
plt.plot(x, np.sin(x), label="sin(x)")  # 绘制单殿图
plt.plot(x, np.cos(x), label = "cos(x)")
# plt.xlim(0, 5)  # X范围
# plt.ylim(-1, 1)  # Y范围
plt.legend() #显示标签
plt.title("welcome ")
plt.xlabel('x')
plt.ylabel('cos or sin')
plt.show()


# scatter plot