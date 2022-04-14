import numpy as np

def analical_solution_withoutPun(X,T):
    K = np.linalg.inv(np.dot(X.T,X))
    W = np.dot(np.dot(K,X.T),T)
    return W

def analical_solution_withPun(X,T,lambd):
    K = np.linalg.inv(np.dot(X.T,X)+lambd*np.identity(X.shape[1]))
    W = np.dot(np.dot(K,X.T),T)
    return W

