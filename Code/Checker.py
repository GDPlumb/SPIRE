
import numpy as np

class Checker:
    def __init__(self, mode, min_change = None):
        self.mode = mode
        self.min_change = min_change
        
    def check(self, y, y_pert):
        if self.mode == "label":
            return np.squeeze(1.0 * (np.rint(y_pert) != np.rint(y)))
        if self.mode == "prob":
            return np.squeeze(1.0 * (np.abs(y_pert - y) >= self.min_change))
