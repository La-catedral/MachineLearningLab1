import numpy as np

#计算单个样本的值
def poly_compute(X,W):
    Y = (np.zeros(X.shape[0])).reshape(-1,1)
    Y = np.dot(X,W)
    return Y

