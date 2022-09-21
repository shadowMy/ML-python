# 多变量线性回归
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gradientDescent import gradientDescent

path = 'ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

# 数据预处理：特征归一化：使用标准差std进行
data2 = (data2 - data2.mean()) / data2.std()

# 代价函数计算和梯度下降算法都是用的矩阵形式运算，因此多变量只不过是X多了几列，不影响矩阵运算。
data2.insert(0, 'Ones', 1)
cols = data2.shape[1]
X2 = data2.iloc[:, 0:cols - 1]
y2 = data2.iloc[:, cols - 1:cols]
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0, 0, 0]))

iters = 1500
alpha = 0.01

theta2, J_history2, theta_history2 = gradientDescent(X2, y2, theta2, alpha, iters)
print('最终的代价函数值为:\n', J_history2[-1])
print('最终的参数theta为：\n', theta2)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), J_history2, 'r')
ax.set_title('Training data with Linear regression with multiple variables')
ax.set_xlabel('Number of iteration')
ax.set_ylabel('Cost J')
plt.show()


# 正规方程
def normalEqn(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y  # X.T@X等价于X.T.dot(X)
    return theta


final_theta2 = normalEqn(X2, y2)  # 感觉和批量梯度下降的theta的值有点差距

