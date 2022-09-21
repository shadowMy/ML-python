import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gradientDescent import gradientDescent
from computeCost import computeCost
from mpl_toolkits import mplot3d

# 导入数据
path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

# 数据可视化展示
'''
print(data.head())  # 显示前五行数据
print(data.describe())
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
plt.show()
'''

# 数据初始化
# 训练集插入一列1作为X0，并将变量X，y初始化
data.insert(0, 'Ones', 1)
cols = data.shape[1]  # shape取1返回列数,取0返回行数；rows = data.shape[0]
X = data.iloc[:, 0:cols - 1]  # iloc
y = data.iloc[:, cols - 1:cols]
# print(X.head()) print(y.head())

# 将DataFrame转换为numpy矩阵以以进行计算，并定义且初始化theta
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))  # 1行2列矩阵，所以在computeCost中需要转置。
alpha = 0.01
iters = 1500

# GO
# 1.测试代价函数
print('Testing the cost function ...')
J = computeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = %.4f' % J)
print('Expected cost value (approx) 32.07\n')

# 2.使用梯度下降计算最终参数theta
print('Running Gradient Descent ...')
theta, J_history, theta_history = gradientDescent(X, y, theta, alpha, iters)
print('Expected theta values (approx) -3.6303,1.1664')
print('迭代%d次，最终的代价函数值为:\n', J_history[-1] % iters)
print('最终的参数theta为：\n', theta)

# 3.预测
predict1 = np.array([1, 3.5]) * theta.T
print('35000人的收益为：', float(predict1 * 10000), '元')
predict2 = np.array([1, 7]) * theta.T
print('70000人的收益为：', float(predict2 * 10000), '元')


#4.拟合可视化及J、theta随迭代次数变化图
'''
#拟合效果图
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = theta[0, 0] + (theta[0, 1] * x)
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', linewidth=3, label='Prediction')
ax.scatter(data.Population, data.Profit, c='c', marker='o', label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population of City in 10,000s')
ax.set_ylabel('Profit in $10,000s')
ax.set_title('Training data with linear regression fit')
plt.show()
#代价函数J变化图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), sharex=True)  #一行二列两张图，分别为ax1、ax2，sharex：共享x坐标
ax1.plot(np.arange(iters), J_history, 'r')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Cost')
ax1.set_title('cost J of each step')
#theta变化图
theta0_history, theta1_history = theta_history[0, :], theta_history[1, :]
ax2.plot(np.arange(iters), theta0_history, 'b', label='theta0')
ax2.plot(np.arange(iters), theta1_history, 'g', label='theta1')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('value of theta')
ax2.legend(loc=2)
ax2.set_title('theta of each step')
plt.show()
'''
#5.观察J随theta0、theta1变化的三维图及等高线图
theta0_vals, theta1_vals = np.linspace(-10, 10, 100), np.linspace(-1, 4, 100)
l0, l1 = len(theta0_vals), len(theta1_vals)
J_vals = np.zeros((l0, l1))
for i in range(l0):
    for j in range(l1):
        temp = np.matrix(np.array([theta0_vals[i], theta1_vals[j]]))
        J_vals[i, j] = computeCost(X, y, temp)

x, y = np.meshgrid(theta0_vals, theta1_vals)
plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(10, 8), dpi=80)
ax3 = plt.gca(projection='3d')
# cmap是颜色映射表
# from matplotlib import cm
# ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = cm.coolwarm)
# cmap = "rainbow" 亦可,改变cmap参数可以控制三维曲面的颜色组合, 一般见到的三维曲面就是rainbow
ax3.plot_surface(x, y, J_vals, rstride=1, cstride=1, cmap=plt.get_cmap('coolwarm'))
# 绘制从3D曲面到底部的投影,zdir 可选 'z'|'x'|'y'| 分别表示投影到z,x,y平面
# zdir = 'z', offset = -2 表示投影到z = -2上
# ax3.contour(x, y, J_vals, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
ax3.set_xlabel('theta0')
ax3.set_ylabel('theta1')
ax3.set_zlabel('cost')
ax3.tick_params(labelsize=10)
ax3.view_init(20, 135)
plt.show()
