
import json
import numpy as np
import pickle
import sys
import torch
import torchvision.models as models

from Config import get_data_dir
from Misc import id_from_path, load_data

sys.path.insert(0, '../Common/')
from Dataset import ImageDataset, my_dataloader
from ModelWrapper import ModelWrapper
from ResNet import get_model

def search(model_dir, data_dir):

    # Load the images for this pair
    with open('{}/splits.json'.format(data_dir), 'rb') as f:
        splits = json.load(f)
    
    with open('{}/images.p'.format(data_dir), 'rb') as f:
        images = pickle.load(f)
        
    # Setup the model
    model = get_model(mode = 'eval', parent = '{}/model.pt'.format(model_dir))
    model.cuda()
    model.eval()
    
    wrapper = ModelWrapper(model, get_names = True)
    
    # Get the model's predictions on each images split
    metrics = {}
    
    def get_map(wrapper, images, ids, name):
        files_tmp, labels_tmp = load_data(ids, images, [name])
        dataset_tmp = ImageDataset(files_tmp, labels_tmp, get_names = True)
        dataloader_tmp = my_dataloader(dataset_tmp)
        y_hat, y_true, names = wrapper.predict_dataset(dataloader_tmp)
        pred_map = {}
        for i in range(len(y_hat)):
            pred_map[id_from_path(names[i])] = (1 * (y_hat[i] >= 0.5))[0]
        return pred_map
        
    def get_diff(map1, map2, min_size = 25):
        n = len(map1)
        if n < min_size:
            return -1
        else:
            changed = 0
            for key in map1:
                if map1[key] != map2[key]:
                    changed += 1
            return changed / n
        
    ids = splits['1s']
    map_orig = get_map(wrapper, images, ids, 'orig')
    for name in ['spurious-box', 'spurious-pixel-paint']:
        map_name = get_map(wrapper, images, ids, name)
        metrics['{} and {}'.format('1s', name)] = get_diff(map_orig, map_name)
        
    ids = splits['1ns']
    map_orig = get_map(wrapper, images, ids, 'orig')
    for name in ['1ns+1s']:
        map_name = get_map(wrapper, images, ids, name)
        metrics['{} and {}'.format('1ns', name)] = get_diff(map_orig, map_name)

    ids = splits['0s']
    map_orig = get_map(wrapper, images, ids, 'orig')
    for name in ['spurious-box', 'spurious-pixel-paint']:
        map_name = get_map(wrapper, images, ids, name)
        metrics['{} and {}'.format('0s', name)] = get_diff(map_orig, map_name)

    ids = splits['0ns']
    map_orig = get_map(wrapper, images, ids, 'orig')
    for name in ['0ns+1s']:
        map_name = get_map(wrapper, images, ids, name)
        metrics['{} and {}'.format('0ns', name)] = get_diff(map_orig, map_name)
        
    with open('{}/search.json'.format(model_dir), 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':

    mode = sys.argv[1]
    label1 = sys.argv[2]
    label2 = sys.argv[3]
    spurious = sys.argv[4]
    trials = sys.argv[5].split(',')

    for trial in trials:
        model_dir = './Models/{}-{}/{}/{}/trial{}'.format(label1, label2, spurious, mode, trial)
        data_dir = '{}/{}-{}/{}/val'.format(get_data_dir(), label1, label2, spurious)
        search(model_dir, data_dir)
