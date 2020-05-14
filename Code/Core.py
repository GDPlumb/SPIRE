
from scipy.special import expit as sigmoid
import numpy as np

def acc(model, X, y):
        pred = model(X).numpy()
        y_hat = 1.0 * (pred >= 0)
        return np.mean(y_hat == y)
        
def prob(model, X):
    return sigmoid(model(X).numpy())
