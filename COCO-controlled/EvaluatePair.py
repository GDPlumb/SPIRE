
import json
import numpy as np
import sys
import torch
import torchvision.models as models

from Misc import get_pair

sys.path.insert(0, '../COCO/')
from COCOWrapper import COCOWrapper
from Dataset import ImageDataset, my_dataloader
from ModelWrapper import ModelWrapper

def evaluate(main, spurious, cop_with_main, cop_without_main, trial, root = '/home/gregory/Datasets/COCO', year = '2017'):

    base = './Pairs/{}-{}/{}-{}/trial{}'.format(main, spurious, cop_with_main, cop_without_main, trial)
    
    # Get the 'testing' images
    coco = COCOWrapper(root = root, mode = 'val', year = year)
    both, just_main, just_spurious, neither = get_pair(coco, main, spurious)

    # Setup the model
    model = models.mobilenet_v2(pretrained = True)
    model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 1)
    model.cuda()
    
    model.load_state_dict(torch.load('{}/model.pt'.format(base)))
    model.eval()
    
    wrapper = ModelWrapper(model)
    
    # Run the evaluation
    def process_set(files, label):
        files_tmp = []
        labels_tmp = []
        for f in files:
            files_tmp.append('{}/val{}/{}'.format(root, year, f))
            labels_tmp.append(np.array([label], dtype = np.float32))
            
        labels_tmp = np.array(labels_tmp, dtype = np.float32)

        dataset_tmp = ImageDataset(files_tmp, labels_tmp)

        dataloader_tmp = my_dataloader(dataset_tmp)

        y_hat, y_true = wrapper.predict_dataset(dataloader_tmp)

        return np.mean(1 * (y_hat >= 0.5) == y_true)
    
    acc_both = process_set(both, 1)
    acc_main = process_set(just_main, 1)
    acc_spur = process_set(just_spurious, 0)
    acc_neither = process_set(neither, 0)
    
    out = {}
    out['both'] = acc_both
    out['just_main'] = acc_main
    out['just_spurious'] = acc_spur
    out['neither'] = acc_neither
    
    out['average'] = np.mean([acc_both, acc_main, acc_spur, acc_neither])
    out['train_dist'] = 0.5 * cop_with_main * acc_both + 0.5 * (1 - cop_with_main) * acc_main + 0.5 * cop_without_main * acc_spur + 0.5 * (1 - cop_without_main) * acc_neither
    
    with open('{}/results.json'.format(base), 'w') as f:
        json.dump(out, f)

if __name__ == '__main__':

    main = sys.argv[1]
    spurious = sys.argv[2]
    cop_with_main = float(sys.argv[3])
    cop_without_main = float(sys.argv[4])
    trials = sys.argv[5].split(',')

    for trial in trials:
        evaluate(main, spurious, cop_with_main, cop_without_main, trial)
