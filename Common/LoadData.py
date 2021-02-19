
from glob import glob
import json
import numpy as np

def load_images(base_dir, subdir_list, include_subdir_name = True):
    
    # Get the original images and labels
    with open('{}/images.json'.format(base_dir), 'r') as f:
        data = json.load(f)
        
    images = {}
    for id in data:
        images[id] = {}
        images[id]['orig'] = data[id]
        
    # Get the various counterfactual versions that are available
    for subdir in subdir_list:
        for cf_dir in glob('{}/{}/*'.format(base_dir, subdir)):
            
            with open('{}/images.json'.format(cf_dir), 'r') as f:
                data = json.load(f)
            
            cf_type = cf_dir.split('/')[-1]
            if include_subdir_name:
                name = '{}-{}'.format(subdir, cf_type)
            else:
                name = cf_type
            
            for id in data:
                images[id][name] = data[id]
    
    return images


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
