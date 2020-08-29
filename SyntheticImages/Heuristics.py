
import numpy as np

def heuristic_1(im, meta, use_strong = False):
    objects = np.zeros((64, 64), dtype = bool)
    for i in range(len(meta)):
        if meta[i] is not None:
            objects = np.logical_or(objects, meta[i])
            
    background = np.logical_not(objects)
    im_new = np.copy(im)
    if use_strong:
        color = 0.5 * np.random.randint(low = 0, high = 2) + np.random.uniform(low = -0.03, high = 0.03)
    else:
        color = np.random.uniform()
    im_new[background] = color
    return (im_new, -1)

def heuristic_2(im, meta, use_strong = False):
    im_new = np.copy(im)
    map_ordered = get_ordered_map(meta, 2)
    color = get_background_color(im, meta)
    im_new[map_ordered] = color
    return (im_new, -1)

def heuristic_3(im, meta, use_strong = False):
    im_new = np.copy(im)
    map_ordered = get_ordered_map(meta, 0)
    if use_strong:
        v = np.random.uniform(low = 0.97, high = 1.0)
        if np.random.uniform() < 0.5:
            color = v
        else:
            color = [v, 0, 0]
    else:
        color = np.random.uniform(size = (1,3))
    im_new[map_ordered] = color
    return (im_new, -1)
    
# Augmentation
def apply_heuristic(X, Y, meta, heuristic):
    n = X.shape[0]
    X_new = []
    Y_new = []
    for i in range(n):
        out = heuristic(X[i], meta[i])
        if out is not None:
            X_new.append(out[0])
            if out[1] != -1:
                Y_new.append(out[1])
            else:
                Y_new.append(np.squeeze(Y[i]))
        else:
            X_new.append(None)
            Y_new.append(None)
    return X_new, Y_new
    
def merge(X, Y, X_aug, Y_aug):
    
    X_new = []
    for x in X_aug:
        if x is not None:
            X_new.append(x)
            
    Y_new = []
    for y in Y_aug:
        if y is not None:
            Y_new.append(y)
            
    X_new = np.float32(np.array(X_new))
    Y_new = np.expand_dims(np.float32(np.array(Y_new)), axis = 1)

    X_aug = np.vstack((X, X_new))
    Y_aug = np.vstack((Y, Y_new))
    
    return X_aug, Y_aug

    
# Helpers

def get_background_color(im, meta):
    objects = np.zeros((64, 64), dtype = bool)
    for i in range(len(meta)):
        if meta[i] is not None:
            objects = np.logical_or(objects, meta[i])
    color = np.mean(im[np.logical_not(objects)])
    return color

def get_ordered_map(meta, index):
    if meta[index] is not None:
        map_ordered = meta[index]
        for i in range(index + 1, len(meta)):
            m_later = meta[i]
            map_ordered = np.logical_and(map_ordered, np.logical_not(m_later))
        return map_ordered
    else:
        return np.zeros((64, 64), dtype = bool)
