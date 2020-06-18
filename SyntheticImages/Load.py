
from sklearn.model_selection import train_test_split
import numpy as np

def load(sample, n, p, n_neutral):

    # Create a training dataset with the spurious correlation
    X = np.zeros((n, 64, 64, 3))
    Y = np.zeros((n, 1))
    meta = []

    for i in range(n):
        x, y, out = sample(p = p)
        X[i, :] = x
        Y[i] = y
        meta.append(out)

    X = np.float32(X) / 255
    Y = np.float32(Y)

    X_train, X_test, Y_train, Y_test, meta_train, meta_test = train_test_split(X, Y, meta, test_size = 0.25)
    X_val, X_test, Y_val, Y_test, meta_val, meta_test = train_test_split(X_test, Y_test, meta_test, test_size = 0.5)

    # Create a neutral dataset without the spurious correlation for evaluation
  
    X_neutral = np.zeros((n_neutral, 64, 64, 3))
    Y_neutral = np.zeros((n_neutral, 1))

    for i in range(n_neutral):
        x, y, out = sample(p = 0.5)
        X_neutral[i, :] = x
        Y_neutral[i] = y

    X_neutral = np.float32(X_neutral) / 255
    Y_neutral = np.float32(Y_neutral)
    
    return  X_train, X_val, X_test, X_neutral, Y_train, Y_val, Y_test, Y_neutral, meta_train, meta_val, meta_test
