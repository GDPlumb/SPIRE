
import numpy as np
from scipy.special import expit as sigmoid
from sklearn.metrics import precision_score, recall_score
import torch

class ModelWrapper():

    def __init__(self, model, get_names = False):
        self.model = model
        self.get_names = get_names
        
    def predict(self, im):
        if len(im.size()) == 3:
            im = torch.unsqueeze(im, 0)
        return sigmoid(self.model(im.cuda()).cpu().data.numpy())
        
    def predict_dataset(self, dataloader):
    
        get_names = self.get_names
    
        y_hat = []
        y_true = []
        if get_names:
            names = []

        for data in dataloader:
            inputs = data[0]
            labels = data[1]
            if get_names:
                for name in data[2]:
                    names.append(name)
            y_hat.append(1.0 * (self.predict(inputs) > 0.5))
            y_true.append(labels.numpy())
        
        y_hat = np.concatenate(np.array(y_hat), axis = 0)
        y_true = np.concatenate(np.array(y_true), axis = 0)
        
        if get_names:
            return y_hat, y_true, names
        else:
            return y_hat, y_true
        
    def metrics(self, y_hat, y_true):
    
        dim = y_hat.shape[1]

        precision = np.zeros((dim))
        for i in range(dim):
            precision[i] = precision_score(np.squeeze(y_true[:, i]), np.squeeze(y_hat[:, i]), zero_division = 0)
            
        recall = np.zeros((dim))
        for i in range(dim):
            recall[i] = recall_score(np.squeeze(y_true[:, i]), np.squeeze(y_hat[:, i]), zero_division = 0)
           
        return precision, recall
