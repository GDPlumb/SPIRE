
import json
import numpy as np
import pickle
import sys
import torch
import torchvision.models as models

from Config import get_data_dir
from Misc import id_from_path, load_data

sys.path.insert(0, '../COCO/')
from Dataset import ImageDataset, my_dataloader
from ModelWrapper import ModelWrapper

def search(mode, main, spurious, p_correct, trial):

    base = './Models/{}-{}/{}/{}/trial{}'.format(main, spurious, p_correct, mode, trial)
    
    # Load the images for this pair
    data_dir = '{}/{}-{}/val'.format(get_data_dir(), main, spurious)
    with open('{}/splits.p'.format(data_dir), 'rb') as f:
        splits = pickle.load(f)
    
    with open('{}/images.p'.format(data_dir), 'rb') as f:
        images = pickle.load(f)
        
    # Setup the model
    model = models.mobilenet_v2(pretrained = True)
    model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 1)
    model.cuda()
    
    model.load_state_dict(torch.load('{}/model.pt'.format(base)))
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
        
    def get_diff(map1, map2):
        n = len(map1)
        changed = 0
        for key in map1:
            if map1[key] != map2[key]:
                changed += 1
        return changed / n
        
    def get_fail(map1, map2):
        detected = 0
        changed = 0
        for key in map1:
            if map1[key] == 1:
                detected += 1
                if map2[key] == 0:
                    changed += 1
        return changed / detected
        
    ids = splits['both']
    map_orig = get_map(wrapper, images, ids, 'orig')
    for name in ['spurious-box', 'spurious-pixel-paint', 'main-box', 'main-pixel-paint', 'both-inverse', 'main-inverse', 'spurious-inverse']:
        map_name = get_map(wrapper, images, ids, name)
        metrics['{} and {}'.format('both', name)] = get_fail(map_orig, map_name)
    
    ids = splits['just_main']
    map_orig = get_map(wrapper, images, ids, 'orig')
    for name in ['main-box', 'main-pixel-paint', 'main-inverse']:
        map_name = get_map(wrapper, images, ids, name)
        metrics['{} and {}'.format('just_main', name)] = get_fail(map_orig, map_name)
        
    ids = splits['just_spurious']
    map_orig = get_map(wrapper, images, ids, 'orig')
    for name in ['spurious-box', 'spurious-pixel-paint', 'spurious-inverse']:
        map_name = get_map(wrapper, images, ids, name)
        metrics['{} and {}'.format('just_spurious', name)] = get_fail(map_orig, map_name)
        
    with open('{}/search.json'.format(base), 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':

    mode = sys.argv[1]
    main = sys.argv[2]
    spurious = sys.argv[3]
    p_correct = float(sys.argv[4])
    trials = sys.argv[5].split(',')

    for trial in trials:
        search(mode, main, spurious, p_correct, trial)
