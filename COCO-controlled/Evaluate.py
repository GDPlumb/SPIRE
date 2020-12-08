
import json
import numpy as np
import pickle
import sys
import torch
import torchvision.models as models

from Config import get_data_dir
from Misc import load_data

sys.path.insert(0, '../COCO/')
from Dataset import ImageDataset, my_dataloader
from ModelWrapper import ModelWrapper

def evaluate(mode, main, spurious, p_correct, trial):

    base = './Models/{}-{}/{}/{}/trial{}'.format(main, spurious, p_correct, mode, trial)
    
    # Load the images for this pair
    data_dir = '{}/{}-{}/val'.format(get_data_dir(), main, spurious)
    with open('{}/splits.p'.format(data_dir), 'rb') as f:
        splits = pickle.load(f)
    
    with open('{}/images.p'.format(data_dir), 'rb') as f:
        images = pickle.load(f)
        
    # Setup the model
    model = models.vgg16(pretrained = True)
    model.classifier[6] = torch.nn.Linear(in_features = 4096, out_features = 1)
    model.cuda()
    
    model.load_state_dict(torch.load('{}/model.pt'.format(base)))
    model.eval()
    
    wrapper = ModelWrapper(model)
        
    # Run the evaluation
    out = {}
    avg = 0
    for name in ['both', 'just_main', 'just_spurious', 'neither']:
        ids = splits[name]
        files_tmp, labels_tmp = load_data(ids, images, ['orig'])
        
        dataset_tmp = ImageDataset(files_tmp, labels_tmp)
        dataloader_tmp = my_dataloader(dataset_tmp)
        
        y_hat, y_true = wrapper.predict_dataset(dataloader_tmp)
        
        v = np.mean(1 * (y_hat >= 0.5) == y_true)
        out[name] = v
        avg += v
    avg /= 4
    out['average'] = avg
    
    with open('{}/results.json'.format(base), 'w') as f:
        json.dump(out, f)

if __name__ == '__main__':

    mode = sys.argv[1]
    main = sys.argv[2]
    spurious = sys.argv[3]
    p_correct = float(sys.argv[4])
    trials = sys.argv[5].split(',')

    for trial in trials:
        evaluate(mode, main, spurious, p_correct, trial)
