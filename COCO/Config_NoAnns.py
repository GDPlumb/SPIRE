import json
import numpy as np
from sklearn.linear_model import LogisticRegression

MAIN = 'tennis+racket'
SPURIOUS = 'person'

def get_pairs():
    return ['{}-{}'.format(MAIN, SPURIOUS)]

def get_object(name):
    with open('./Categories.json', 'r') as f:
        cats = json.load(f)
    for cat in cats:
        if cat['name'] == name.replace('+', ' '):
            index = int(cat['id'])
            break

    return name, index

def get_main():
    return get_object(MAIN)

def get_spurious():
    return get_object(SPURIOUS)

def get_confidence():
    return 0.0001

def get_sampling_prob():
    return 1.0

# Add or remove Spurious by projecting across a linear classifier that detects Spurious
def project(rep):
    confidence = get_confidence()
    _, index_m = get_main()
    _, index_s = get_spurious()

    x = []
    y = []
    ids = list(rep)
    for i in ids:
        data = rep[i]
        x.append(data[0])
        y.append(data[1])
    x = np.array(x)
    y = np.array(y)

    lm = LogisticRegression(random_state = 0, solver = 'sag', max_iter = 1000).fit(x, y[:, index_s])

    def compare_low(t):
        return t > confidence
    
    def compare_high(t):
        return t < 1 - confidence            
            
    cf = {}
    for i in ids:
        x_tmp = rep[i][0]
        y_tmp = rep[i][1]
        
        if y_tmp[index_s] == 1:
            comp = compare_low
            scale = -0.1
        else:
            comp = compare_high
            scale = 0.1
            
        x_tmp = np.expand_dims(np.copy(x_tmp), 0)
        
        while True:
            prob = lm.predict_proba(x_tmp)[0, index_s]
            
            if comp(prob):
                x_tmp += scale * lm.coef_
            else:
                y_tmp = np.copy(y_tmp)
                y_tmp[index_s] = 1 - y_tmp[index_s]
                cf[i] = [np.squeeze(x_tmp), y_tmp]
                break  
                
    return cf
