
import json
import numpy as np

from Config import get_data_dir

def get_sc_strength(data):
    return data['s_given_m'] - data['s_given_not_m']

def get_cf_score(data):
    return data['cf_score']

def get_d_gap(data):
    v_b = data['both']
    v_m = data['just_main']
    if v_b == -1 or v_m == -1:
        return -1
    else:
        return data['both'] - data['just_main']

def get_h_gap(data):
    v_s = data['just_spurious']
    v_n = data['neither']
    if v_s == -1 or v_n == -1:
        return -1
    else:
        return data['neither'] - data['just_spurious'] 


def load_data_fs(ids, images):
    files = []
    labels = []
    contexts = [] #1 -> spurious (ie, in context), 0 -> no spurious (ie, out of context)
    
    with open('./FindSCs.json', 'r') as f:
        pairs = json.load(f)
    
    ids_context = []
    spurious_processed = []
    for pair in pairs:
        spurious = pair.split('-')[1]
        
        if spurious not in spurious_processed:
            with open('{}/train/splits/{}.json'.format(get_data_dir(), pair)) as f:
                splits = json.load(f)
            
            for id in splits['both']:
                ids_context.append(id)
                
            for id in splits['just_spurious']:
                ids_context.append(id)
        
            spurious_processed.append(spurious)
            
    ids_context = set(ids_context)
    
    for id in ids:
        img = images[id]['orig']
        files.append(img[0])
        labels.append(img[1])
        if id in ids_context:
            contexts.append(1)
        else:
            contexts.append(0)

    labels = np.array(labels, dtype = np.float32)
    labels = labels.reshape((labels.shape[0], 91))
    
    contexts = np.array(contexts, dtype = np.float32)
    contexts = contexts.reshape((contexts.shape[0], 1))
    
    return files, labels, contexts
