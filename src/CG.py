import numpy as np

def co_gradient_withoutPun(X,T,W):
    A = np.dot(X.T,X) # X (m,n)
    b = np.dot(X.T,T)
    X = W
    r = b - np.dot(A,X)
    p = r
    k = 0
    while True:
        alpha = np.dot(r.T,r)/np.dot(np.dot(p.T,A),p)
        X = X + alpha*p
        k += 1
        r_next = r - alpha* np.dot(A,p)
        if np.linalg.norm(r_next) < 0.0000001:
            break
        beta = np.dot(r_next.T,r_next)/np.dot(r.T,r)
        p = r_next + beta * p
        r = r_next
    print("迭代了"+str(k)+"次")
    return X,k

def co_gradient_withPun(X,T,W,lambd):
    A = np.dot(X.T, X)+lambd*np.identity(X.T.shape[0])
    b = np.dot(X.T, T)
    X = W #上面三行将参数代入线性方程组
    r = b - np.dot(A, X)
    p = r
    k = 0
    while True:
        alpha = np.dot(r.T, r)/np.dot(np.dot(p.T, A), p) #计算步长
        X = X + alpha * p #对参数W，也就是这里的X进行迭代
        k += 1
        r_next = r - alpha * np.dot(A, p) #计算下一个残差，后面还用到当前的残差，故需要保留
        if np.linalg.norm(r_next) < 0.000001:
            break #当残差小于某个值我们认为拟合效果足够，停止迭代
        beta = np.dot(r_next.T, r_next) / np.dot(r.T, r) #计算搜索方向式子中需要的参数
        p = r_next + beta * p #更新搜索方向
        r = r_next #更新残差
    print("迭代了"+str(k)+"次")
    return X,k
