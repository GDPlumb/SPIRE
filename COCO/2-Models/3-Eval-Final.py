
from collections import defaultdict
import json
import pickle
import numpy as np
from sklearn.metrics import average_precision_score
import sys

from Worker import get_metrics

sys.path.insert(0, '../')
from Config import get_data_dir

def get_map(preds, data_split = 'val'):
    
    # Get the labels for the dataset
    with open('{}/{}/images.json'.format(get_data_dir(), data_split), 'r') as f:
        images = json.load(f)
    ids = list(images)
    
    # Collect the predictions and labels
    y_hat = []
    y_true = []
    for i in ids:
        y_hat.append(preds['orig'][i])
        y_true.append(images[i][1])
    y_hat = np.array(y_hat)
    y_true = np.array(y_true)
    
    # Calculated MAP
    with open('./Categories.json', 'r') as f:
        cats = json.load(f)
        
    v = []
    for cat in cats:
        i = cat['id']
        v.append(average_precision_score(y_true[:, i], y_hat[:, i]))
    
    return np.mean(v)
    
 
modes = ['initial-tune', 'spire', 'fs-3', 'na-transfer']
trials = [0,1,2,3,4,5,6,7]

data_split = 'val'
max_samples = None
 
for mode in modes:
    for trial in trials:

        model_dir = './Models/{}/trial{}'.format(mode, trial)

        with open('{}/pred.pkl'.format(model_dir), 'rb') as f:
            preds = pickle.load(f)


        with open('../0-FindPairs/Pairs.json', 'r') as f:
            pairs = json.load(f)

        out = {}
        for pair in pairs:
            info = get_metrics(pair, preds, data_split = data_split, max_samples = max_samples)
            out[pair] = info
            
        out['MAP'] = get_map(preds, data_split = data_split)

        with open('{}/results.pkl'.format(model_dir), 'wb') as f:
            pickle.dump(out, f)