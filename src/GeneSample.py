import numpy as np
import random

def gene_sample(sta,end,numOfPoint):
    X = np.linspace(sta,end,numOfPoint,endpoint = True)
    T = np.sin(2*np.pi*X)
    mu = 0 #mu and sigma of gauss distribution
    sigma = 0.2
    for i in range(X.size):
        # X[i] += random.gauss(mu,sigma)
        T[i] += random.gauss(mu,sigma)
    return X.reshape(X.size,1),T.reshape(T.size,1)

def gene_standard_sample(sta,end,numOfPoint):
    X = np.linspace(sta,end,numOfPoint,endpoint = True)
    T = np.sin(2*np.pi*X)
    return X.reshape(X.size,1),T.reshape(T.size,1)
