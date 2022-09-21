# 梯度下降函数theta = theta - alpha/m * ((X*theta - y)' * X)'

import numpy as np
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, iters):  # alpha学习率，iters迭代次数
    m = len(y)
    '''temp = np.matrix(np.zeros(theta.shape))
    #parameters = int(theta.ravel().shape[1])'''
    J_history = np.zeros(iters)     # 使用 iters*1维矩阵，保存theta每次迭代的历史
    theta_history = np.zeros([X.shape[1], iters])   # 使用 2*iters维矩阵，保存theta每次迭代的历史

    for i in range(iters):
        #error h(x)-y  m*1维
        error = (X * theta.T) - y
        '''for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))'''
        #sum((h(x)-y)x) 1*2维
        lineLope = error.T * X
        theta = theta - (alpha / m) * lineLope

        J_history[i] = computeCost(X, y, theta)
        theta_history[:, i] = theta
    return theta, J_history, theta_history
