
import numpy as np
import os
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
import sys
import torch
import torchvision.models as models

from Config import get_data_dir
from Misc import load_data

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

    base = './Models/{}-{}/{}/{}/trial{}'.format(main, spurious, p_correct, mode, trial)
    os.system('rm -rf {}'.format(base))
    Path(base).mkdir(parents = True, exist_ok = True)

    name = '{}/model'.format(base)
    
    # Load the chosen images for this pair
    data_dir = '{}/{}-{}/train'.format(get_data_dir(), main, spurious)
    with open('{}/splits.p'.format(data_dir), 'rb') as f:
        splits = pickle.load(f)
    both = splits['both']
    just_main = splits['just_main']
    just_spurious = splits['just_spurious']
    neither = splits['neither']
    
    with open('{}/images.p'.format(data_dir), 'rb') as f:
        images = pickle.load(f)
    
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
    
    ids = []
    for id in both_final:
        ids.append(id)
    for id in just_main_final:
        ids.append(id)
    for id in just_spurious_final:
        ids.append(id)
    for id in neither_final:
        ids.append(id)
        
    # Train/validation split for the ids
    # By splitting on Image ID, we ensure all counterfactuals are in the same fold
    ids_train, ids_val = train_test_split(ids, test_size = 0.1)

    # Load the the data specified by mode for each Image ID
    if mode in ['initial-transfer', 'initial-tune']:
        names = ['orig']
    elif mode in ['both-transfer', 'both-tune']:
        names = ['orig', 'box-main', 'box-spurious']
    elif mode in ['spurious-transfer', 'spurious-tune']:
    	names = ['orig', 'box-spurious']
    elif mode in ['spurious-paint-transfer', 'spurious-paint-tune']:
    	names = ['orig', 'pixel-spurious-paint']
        
    files_train, labels_train = load_data(ids_train, images, names)
    files_val, labels_val = load_data(ids_val, images, names)

    datasets = {}
    datasets['train'] = ImageDataset(files_train, labels_train)
    datasets['val'] = ImageDataset(files_val, labels_val)

    dataloaders = {}
    dataloaders['train'] = my_dataloader(datasets['train'])
    dataloaders['val'] = my_dataloader(datasets['val'])
    
    # Setup the model and optimization process
    model = models.mobilenet_v2(pretrained = True)

    if 'transfer' in mode.split('-'):
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 1)
        optim_params = model.classifier.parameters()
        lr = 0.001
    elif 'tune' in mode.split('-'):
        model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 1)
        model.load_state_dict(torch.load('./Models/{}-{}/{}/initial-transfer/trial{}/model.pt'.format(main, spurious, p_correct, trial)))
        optim_params = model.parameters()
        lr = 0.0001

    model.cuda()

    metric_loss = torch.nn.BCEWithLogitsLoss()
    
    model = train_model(model, optim_params, dataloaders, metric_loss, metric_acc_batch, metric_acc_agg, name = name,
                        lr_init = lr, select_cutoff = 5, decay_max = 1)
    torch.save(model.state_dict(), '{}.pt'.format(name))

if __name__ == '__main__':
    
    mode = sys.argv[1]
    main = sys.argv[2]
    spurious = sys.argv[3]
    p_correct = float(sys.argv[4])
    trials = sys.argv[5].split(',')
    
    for trial in trials:
        train(mode, main, spurious, p_correct, trial)
