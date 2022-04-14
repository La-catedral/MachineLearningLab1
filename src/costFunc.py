import numpy as np

#输入 存储n个样本预测结果的向量Y 和 n个样本给定标签的向量T 计算损失

#惩罚待输入
def cost_func_pun(Y,T,lambd,W):
    a =  1/2 * np.dot((Y-T).T,Y-T)+ lambd*np.dot(W.T,W)
    return a[0]

def cost_func_withoutPun(Y,T):
    a = 1/2 * np.dot((Y-T).T,Y-T)
    return a[0]



