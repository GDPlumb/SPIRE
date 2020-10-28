
import numpy as np
import os
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
import sys
import torch
import torchvision.models as models

sys.path.insert(0, '../COCO/')
from Dataset import ImageDataset, my_dataloader

sys.path.insert(0, '../Common/')
from Train_info import train_model

def metric_acc_batch(y_hat, y):
    y_hat = y_hat.cpu().data.numpy()
    y_hat = 1 * (y_hat >= 0)
    y = y.cpu().data.numpy()

    out = np.zeros((2))
    out[0] = np.sum(y == y_hat)
    out[1] = len(y)
    
    return out

def metric_acc_agg(counts_list = None):
    if counts_list is None:
        return ['Acc']
    else:
        correct = 0
        total = 0
        
        for counts in counts_list:
            correct += counts[0]
            total += counts[1]
            
        return [correct / total]
        
def train(mode, main, spurious, p_correct, trial, p_main = 0.5, p_spurious = 0.5, n = 2000):

    parent = './Pairs/{}-{}'.format(main, spurious)
    
    base = '{}/{}/{}/trial{}'.format(parent, p_correct, mode, trial)
    os.system('rm -rf {}'.format(base))
    Path(base).mkdir(parents = True, exist_ok = True)

    name = '{}/model'.format(base)
    
    # Load the chosen images for this pair
    with open('{}/splits.p'.format(parent), 'rb') as f:
        splits = pickle.load(f)
    both = splits[0]
    just_main = splits[1]
    just_spurious = splits[2]
    neither = splits[3]
    
    # Find the number of images to get from each split
    num_main = int(n * p_main)
    num_spurious = int(n * p_spurious)

    num_both = int(p_correct * num_spurious)
    num_just_main = num_main - num_both
    num_just_spurious = num_spurious - num_both
    num_neither = n - num_both - num_just_main - num_just_spurious

    if num_both < 0 or num_just_main < 0 or num_just_spurious < 0 or num_neither < 0:
        print('Error: Bad Distribution Setup')
        print(num_both, num_just_main, num_just_spurious, num_neither)
        sys.exit(0)
    
    both_final = both[:num_both]
    just_main_final = just_main[:num_just_main]
    just_spurious_final = just_spurious[:num_just_spurious]
    neither_final = neither[:num_neither]

    # Setup the train/validation split and data loaders
    files = []
    labels = []

    for f in both_final:
        files.append(f)
        labels.append(np.array([1], dtype = np.float32))
        
    for f in just_main_final:
        files.append(f)
        labels.append(np.array([1], dtype = np.float32))
        
    for f in just_spurious_final:
        files.append(f)
        labels.append(np.array([0], dtype = np.float32))

    for f in neither_final:
        files.append(f)
        labels.append(np.array([0], dtype = np.float32))
        
    labels = np.array(labels, dtype = np.float32)

    files_train, files_val, labels_train, labels_val = train_test_split(files, labels, test_size = 0.1)

    datasets = {}
    datasets['train'] = ImageDataset(files_train, labels_train)
    datasets['val'] = ImageDataset(files_val, labels_val)

    dataloaders = {}
    dataloaders['train'] = my_dataloader(datasets['train'])
    dataloaders['val'] = my_dataloader(datasets['val'])
    
    # Setup the model and optimization process
    model = models.mobilenet_v2(pretrained = True)

    if mode in ['initial-transfer']:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 1)
        optim_params = model.classifier.parameters()

    model.cuda()

    metric_loss = torch.nn.BCEWithLogitsLoss()
    
    model = train_model(model, optim_params, dataloaders, metric_loss, metric_acc_batch, metric_acc_agg, name = name,
                        select_cutoff = 5, decay_max = 1)
    torch.save(model.state_dict(), '{}.pt'.format(name))

if __name__ == '__main__':
    
    mode = sys.argv[1]
    main = sys.argv[2]
    spurious = sys.argv[3]
    p_correct = float(sys.argv[4])
    trials = sys.argv[5].split(',')
    
    for trial in trials:
        train(mode, main, spurious, p_correct, trial)