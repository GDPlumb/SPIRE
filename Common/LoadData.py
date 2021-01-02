
import numpy as np

def load_data(ids, images, names):
    files = []
    labels = []
    for id in ids:
        for name in images[id]:
            if name in names:
                if type(names) == dict:
                    if np.random.uniform() <= names[name]:
                        files.append(images[id][name][0])
                        labels.append(images[id][name][1])                        
                else:
                    files.append(images[id][name][0])
                    labels.append(images[id][name][1])
    
    labels = np.array(labels, dtype = np.float32)
    
    return files, labels
