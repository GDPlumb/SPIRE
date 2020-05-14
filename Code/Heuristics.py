
import numpy as np

from Core import *

class CategoricalPerturber:
    def __init__(self, encoder = None, decoder = None):
        self.encoder = encoder
        self.decoder = decoder
    
    def apply(self, model, X, h):
    
        if self.encoder is None:
            X_pert = np.copy(X)
            X_pert[:, h[0]] = h[1]
            X_pert = np.float32(X_pert)
            y_pert = prob(model, X_pert)
            return X_pert, y_pert
            
        else:
            rep = self.encoder(X).numpy()
            rep[:, h[0]] = h[1]
            X_pert = self.decoder(rep).numpy()
            X_pert = np.float32(X_pert)
            y_pert = prob(model, X_pert)
            return X_pert, y_pert
            
class ContinuousPerturber:
    def __init__(self, encoder = None, decoder = None):
        self.encoder = encoder
        self.decoder = decoder
        
    def apply(self, model, X, h):
    
        if self.encoder is None:
            X_pert = np.copy(X) + h
            X_pert = np.float32(X_pert)
            y_pert = prob(model, X_pert)
            return X_pert, y_pert
            
        else:
            rep = self.encoder(X).numpy() + h
            X_pert = self.decoder(rep).numpy()
            X_pert = np.float32(X_pert)
            y_pert = prob(model, X_pert)
            return X_pert, y_pert
