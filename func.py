import numpy as np

def _norm(x, Z=True):
    x_mu = np.mean(x, axis=0)
    if Z:
        x_std = np.std(x, axis=0)
        x_n = (x-x_mu)/x_std
    else:
        x_min = np.min(x, axis=0)
        x_max = np.max(x, axis=0)
        x_n = (x-x_mu)/(x_max-x_min)
    return x_n

def _onehot(x):
    A = np.zeros((len(x), 43))
    for i in range(len(x)):
        idx = x[i]
        A[i, idx] = 1
    return A