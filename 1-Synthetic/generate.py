
import numpy as np

# switch to A & B or B & C (make A and C very correlated)
def gen_logic(n):
    X = np.random.uniform(size=(n, 2))
    X = np.hstack((X, X))

    y = np.zeros((n, 1))
    for i in range(n):
        if X[i,0] > 0.5 and X[i, 1] > 0.5:
            y[i] = 1

    return X, y



