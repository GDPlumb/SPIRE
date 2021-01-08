
import numpy as np

def load_data(ids, images, names, indices_preserve = None):
    files = []
    labels = []
    for id in ids:
        for name in images[id]:
            if name in names:
                if type(names) == dict:
                    if np.random.uniform() <= names[name]:
                        files.append(images[id][name][0])
                        
                        label = np.copy(images[id][name][1]) #Copy in case this is somehow linked
                        if indices_preserve is not None:
                            for i in range(len(label)):
                                if i not in indices_preserve:
                                    label[i] = 0.0
                        
                        labels.append(label)
                else:
                    files.append(images[id][name][0])
                    labels.append(images[id][name][1])
    
    labels = np.array(labels, dtype = np.float32)
    
    return files, labels
