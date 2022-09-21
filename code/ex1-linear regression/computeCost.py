# 单变量代价函数
import numpy as np

def computeCost(X, y, theta):
    m = len(y)
    predictions = X * theta.T
    sqrErrors = np.power((predictions - y), 2)
    J = np.sum(sqrErrors) / (2 * m)
    return J
