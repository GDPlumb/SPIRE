
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
    
    with open('./COCO_cats.json', 'r') as f: #This is a json copy of coco.loadCats(coco.getCatIds())
        cats = json.load(f)
        
    with open('./FindSCs.json', 'r') as f:
        pairs = json.load(f)
    
    id_map = {}
    for pair in pairs:
        main = pair.split('-')[0].replace('+', ' ')
        index = None
        for cat in cats:
            if cat['name'] == main:
                index = int(cat['id'])

        with open('{}/train/splits/{}.json'.format(get_data_dir(), pair)) as f:
            splits = json.load(f)
            
        num_both = len(splits['both'])
        num_main = len(splits['just_main'])
        
        if num_both >= num_main:
            split_suppress = 'just_main'
            alpha = np.sqrt(num_both / num_main)
        else:
            split_suppress = 'both'
            alpha = np.sqrt(num_main / num_both)
        if alpha < 20.0:
            alpha = 20.0

        for id in splits[split_suppress]:
            info = (index, alpha)
            if id in id_map:
                id_map[id].append(info)
            else:
                id_map[id] = [info]
                
    files = []
    labels = []
    contexts = [] #0 -> do not supress, !0 -> supress and use that weight
    for id in ids:
        img = images[id]['orig']
        files.append(img[0])
        labels.append(img[1])
        c = np.zeros((91))
        if id in id_map:
            for info in id_map[id]:
                index = info[0]
                alpha = info[1]
                c[index] = alpha
        contexts.append(c)

    labels = np.array(labels, dtype = np.float32)
    labels = labels.reshape((labels.shape[0], 91))
    
    contexts = np.array(contexts, dtype = np.float32)
    contexts = contexts.reshape((contexts.shape[0], 91))
    
    return files, labels, contexts
