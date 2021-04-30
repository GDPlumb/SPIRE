
import json
import numpy as np
import os
import pickle
import sys

from Worker import get_representation

sys.path.insert(0, '../')
from Config import get_data_dir

sys.path.insert(0, '../../Common/')
from LoadData import load_ids
from ResNet import get_model

if __name__ == '__main__':
    
    # Setup the dataset    
    with open('{}/train/images.json'.format(get_data_dir()), 'r') as f:
        images = json.load(f)
    ids = list(images)
    
    filenames, labels = load_ids(ids, images)
    labels = np.array(labels, dtype = np.float32)
    
    # Setup the pretrained model
    model = get_model(mode = 'eval', parent = 'pretrained')
    
    # Get its representation of the dataset
    data = get_representation(model, filenames, labels)
    
    # Save the output
    save_dir = './Models/pretrained'
    os.system('rm -rf {}'.format(save_dir))
    os.system('mkdir {}'.format(save_dir))
    with open('{}/rep.pkl'.format(save_dir), 'wb') as f:
        pickle.dump(data, f)
    