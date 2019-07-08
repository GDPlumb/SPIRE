
import numpy as np

class LowDensity():

    def __init__(self):
        pass
        
    def gen(self, n):
        n_high = np.int(np.round(0.9 * n))
        n_low = n - n_high
        x_high = np.random.uniform(size = (n_high, 2), low = 0.0, high = 0.5)
        x_low =  np.random.uniform(size = (n_low, 2), low = 0.5, high = 1.0)
        return np.vstack((x_high, x_low))
    
    def label(self, X, noise = 0.1):
        n = X.shape[0]
        y = np.zeros((n,1))
        for i in range(n):
            y[i] = 10 * ((X[i, 0] - 0.0) ** 2 + noise * np.random.normal())
        return y
