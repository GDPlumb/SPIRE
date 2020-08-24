
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

def augment(X, Y, meta, heuristic):

    X_aug = np.copy(X)
    Y_aug = np.copy(Y)

    X_new = []
    Y_new = []

    n = X_aug.shape[0]

    for i in range(n):

        im = X_aug[i]
        m = meta[i]

        out = heuristic(im, m)

        if out is not None:
            X_new.append(out[0])
            
            if out[1] != -1:
                Y_new.append(out[1])
            else:
                Y_new.append(np.squeeze(Y_aug[i]))
    

    X_new = np.float32(np.array(X_new))
    Y_new = np.expand_dims(np.float32(np.array(Y_new)), axis = 1)

    X_aug = np.vstack((X_aug, X_new))
    Y_aug = np.vstack((Y_aug, Y_new))
    
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
        print("bounce")
        return np.zeros((64, 64), dtype = bool)
