import numpy as np
import Polycompute
import costFunc
import GeneSample
#学习率alpha待确定


def gradient_decent_withoutPun(X_head,T,W,alpha,num_ite):
    cost = []
    test = []
    Xt,Tt =GeneSample.gene_standard_sample(0,1,100)
    X_headt = np.c_[Xt ** 0, Xt]
    for i in range(2, len(W)):
        X_headt = np.c_[X_headt, Xt ** i]

    for i in range(num_ite):
        K = np.dot(X_head.T, X_head)
        dW = np.dot(K , W) - np.dot(X_head.T, T)
        W = W - alpha * dW
        if i%10 == 0:
            Y = Polycompute.poly_compute(X_head, W)
            cost.append(costFunc.cost_func_withoutPun(Y,T))
            Yt = Polycompute.poly_compute(X_headt,W)
            test.append(costFunc.cost_func_withoutPun(Yt,Tt))
    return W,cost,test

def gradient_decent_withPun(X_head,T,W,alpha,num_ite,lambd):
    cost = []
    test = []
    Xt,Tt= GeneSample.gene_standard_sample(0, 1, 100)
    X_headt = np.c_[Xt ** 0, Xt]
    for i in range(2, len(W)):
        X_headt = np.c_[X_headt, Xt ** i]

    for i in range(num_ite):
        # K = np.dot(W.T, X_head)
        # dW = np.sum(X_head * (K - T), axis=1) + lambd * W
        K = np.dot(X_head.T,X_head)
        dW = np.dot(K  ,W) -np.dot(X_head.T,T) + lambd*W
        W = W - alpha * dW
        if i % 10 == 0:
            Y = Polycompute.poly_compute(X_head, W)
            cost.append(costFunc.cost_func_pun(Y, T,lambd,W))
            Yt = Polycompute.poly_compute(X_head, W)
            test.append(costFunc.cost_func_withoutPun(Yt, Tt,lambd,W))
    return W, cost,test









