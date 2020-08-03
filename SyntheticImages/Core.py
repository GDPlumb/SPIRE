
import math
import numpy as np
from scipy.special import expit as sigmoid

def acc(model, X, y, batch_size = 64):
    n = X.shape[0]
    n_batches = math.ceil(n / batch_size)
    pred = np.zeros((n, 1))
    for i in range(n_batches):
        start = i * batch_size
        stop = min(start + batch_size, n + 1)
        pred[start:stop] = model(X[start:stop]).numpy()
    y_hat = 1.0 * (pred >= 0)
    return np.mean(y_hat == y)
        
def prob(model, X):
    return sigmoid(model(X).numpy())
