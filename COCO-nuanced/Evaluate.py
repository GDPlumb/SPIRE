
import glob
import json
import numpy as np
import pickle
import sys
import torch
import torchvision.models as models

from Config import get_data_dir
from Misc import load_data

sys.path.insert(0, '../Common/')
from Dataset import ImageDataset, my_dataloader
from ModelWrapper import ModelWrapper
from ResNet import get_model

def evaluate(model_dir, data_dir, min_size = 25, challenge_info = None):

    # Load the images for this pair
    with open('{}/splits.json'.format(data_dir), 'rb') as f:
        splits = json.load(f)
    
    with open('{}/images.p'.format(data_dir), 'rb') as f:
        images = pickle.load(f)
        
    # Setup the model
    model = get_model(mode = 'eval', parent = '{}/model.pt'.format(model_dir))
    model.cuda()
    model.eval()
    
    wrapper = ModelWrapper(model)
        
    # Run the evaluation
    out = {}
    for name in ['1s', '1ns', '0s', '0ns']:
        ids = splits[name]
        files_tmp, labels_tmp = load_data(ids, images, ['orig'])
        
        if len(files_tmp) < min_size:
            v = -1
        else:
            dataset_tmp = ImageDataset(files_tmp, labels_tmp)
            dataloader_tmp = my_dataloader(dataset_tmp)
            
            y_hat, y_true = wrapper.predict_dataset(dataloader_tmp)
            
            v = np.mean(1 * (y_hat >= 0.5) == y_true)
        out[name] = v

    # Run the Challenge Set evaluation
    if challenge_info is not None:
        label1 = challenge_info[0]
        label2 = challenge_info[1]
        spurious = challenge_info[2]
        
        avg = 0.0
        for config in [('{}+{}'.format(label1, spurious), 1, 'c-1s'), ('{}-{}'.format(label1, spurious), 1, 'c-1ns'), ('{}+{}'.format(label2, spurious), 0, 'c-0s'), ('{}-{}'.format(label2, spurious), 0, 'c-0ns')]:
            folder = config[0]
            label = config[1]
            name = config[2]

            files_tmp = []
            labels_tmp = []
            for file in glob.glob('./ChallengeSets/{}/*'.format(folder)):
                files_tmp.append(file)
                labels_tmp.append(label)

            dataset_tmp = ImageDataset(files_tmp, labels_tmp)
            dataloader_tmp = my_dataloader(dataset_tmp)

            y_hat, y_true = wrapper.predict_dataset(dataloader_tmp)
            v = np.mean(1 * (y_hat >= 0.5) == y_true)
            avg += v
                    
            out[name] = v
        out['cs-avg'] = avg / 4
    
    
    with open('{}/results.json'.format(model_dir), 'w') as f:
        json.dump(out, f)

if __name__ == '__main__':

    mode = sys.argv[1]
    label1 = sys.argv[2]
    label2 = sys.argv[3]
    spurious = sys.argv[4]
    trials = sys.argv[5].split(',')

    for trial in trials:
        model_dir = './Models/{}-{}/{}/{}/trial{}'.format(label1, label2, spurious, mode, trial)
        data_dir = '{}/{}-{}/{}/val'.format(get_data_dir(), label1, label2, spurious)
        evaluate(model_dir, data_dir, challenge_info = (label1, label2, spurious))
