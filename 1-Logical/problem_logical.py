
import numpy as np

class Logical():

    def __init__(self):
        pass
        
    def gen(self, n):
        return np.random.uniform(size = (n,2))
    
    def gen_bad(self, n):
        X = self.gen(n)
        X_pert = X + np.random.uniform(low = -0.05, high = 0.05, size = X.shape)
        return np.hstack((X,X_pert))
    
    def gen_zeros(self, n):
        X = self.gen(n)
        return np.hstack((X, np.zeros((n,2))))
    
    def gen_random(self, n):
        X = self.gen(n)
        return np.hstack((X, self.gen(n)))
    
    def label(self, X):
        n = X.shape[0]
        y = np.zeros((n,1))
        for i in range(n):
            if X[i, 0] > 0.5 and X[i, 1] > 0.5:
                y[i] = 1.0
        return y
