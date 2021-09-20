
'''
TODOs:
-  These functions are copies of the ones found in COCO/2-Models/Worker.py
-  Refactor that code to use this
-  Refactor the analysis for COCO-nuanced to use this as well (will require changing split names)
'''

import numpy as np
from sklearn.metrics import auc

def get_accs(preds, num = 101):
    thresholds = np.linspace(0, 1, num = num)
    
    accs = {}
    for name in preds:
        POS = None
        if name in ['both', 'just_main']:
            POS = True
        elif name in ['just_spurious', 'neither']:
            POS = False
        else:
            print('Warning:  bad name')
        
        p = preds[name]
        n = len(p)
        p = np.sort(p, axis = 0)        
        
        result = np.zeros((num))
        index = 0
        for i, t in enumerate(thresholds):
            while index < n and p[index] <= t:
                index += 1
            if POS:
                result[i] = 1 - index / n
            else:
                result[i] = index / n
        
        accs[name] = result
    
    return accs

def get_gaps(accs, num = 101):
    thresholds = np.linspace(0, 1, num = num)
        
    r_gap = np.abs(accs['both'] - accs['just_main'])
    h_gap = np.abs(accs['just_spurious'] - accs['neither'])
    
    out = {}
    out['r-gap'] = r_gap
    out['h-gap'] = h_gap
    return out

def get_pr(accs, P_m, P_s_m, P_s_nm):
    tp = P_m * (P_s_m * accs['both'] + (1 - P_s_m) * accs['just_main'])
    fp = (1 - P_m) * (P_s_nm * (1 - accs['just_spurious']) + (1 - P_s_nm) * (1 - accs['neither'])) 
        
    recall = tp / P_m
    precision = tp / (tp + fp + 1e-16)
    precision[np.where(tp == 0.0)] = 1.0
        
    out = {}
    out['precision'] = precision
    out['recall'] = recall
    return out

def interpolate(x, y, x_t):
    y_rev = list(reversed(list(y)))
    x_rev = list(reversed(list(x)))
    return np.interp(x_t, x_rev, y_rev)

def average_same_x(x, y_list):
    n = len(x)
    d = len(y_list)
    out = []
    for i in range(n):
        v = 0.0
        for j in range(d):
            v += y_list[j][i]
        v /= d
        out.append(v)
    return out

def get_ap(pr, num = 101):
    thresholds = np.linspace(0, 1, num = num)
    pr_curve = interpolate(pr['recall'], pr['precision'], thresholds)
    ap = auc(thresholds, pr_curve)
    return ap