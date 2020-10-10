
import json
from sklearn.model_selection import train_test_split
import numpy as np
from pycocotools.coco import COCO
import sys
import torch
import torchvision.models as models

from Dataset import merge_sources, unpack_sources, ImageDataset, my_dataloader

sys.path.insert(0, '../Common/')
from Train_info import train_model

def run(config):
    print(config)
    
    root = config[0]
    year = config[1]
    trial = config[2]
    task = config[3]
    
    # Setup the sources for the datasets and the intial model parameters
    initial_model = None
    if task in ['initial-transfer', 'initial-tune']:
        sources = []
        sources.append('{}/{}{}-info.p'.format(root, 'train', year))
        
        if task != 'initial-transfer':
            initial_model = 'initial-transfer/model_{}.pt'.format(trial)
            
    if task in ['random-tune']:
        sources = []
        sources.append('{}/{}{}-info.p'.format(root, 'train', year))
        sources.append('{}/{}{}-random-info.p'.format(root, 'train', year))
        
        initial_model = 'initial-transfer/model_{}.pt'.format(trial)
            
    if task in ['random-tune-paint']:
        sources = []
        sources.append('{}/{}{}-info.p'.format(root, 'train', year))
        sources.append('{}/{}{}-random-paint-info.p'.format(root, 'train', year))
        
        initial_model = 'initial-transfer/model_{}.pt'.format(trial)
    
    # Unpack the sources and divide the data into training and validation datasets
    # Note:  We ensure that all versions of the same image are in the same fold
    file_dict = merge_sources(sources)
    keys = [key for key in file_dict]
    keys_train, keys_val = train_test_split(keys, test_size = 0.1)
    filenames_train, labels_train = unpack_sources(file_dict, keys = keys_train)
    filenames_val, labels_val =  unpack_sources(file_dict, keys = keys_val)

    datasets = {}
    datasets['train'] = ImageDataset(filenames_train, labels_train)
    datasets['val'] = ImageDataset(filenames_val, labels_val)
    
    dataloaders = {}
    dataloaders['train'] = my_dataloader(datasets['train'])
    dataloaders['val'] = my_dataloader(datasets['val'])
    
    # Setup the model (initial parameters and what parameters are trainable)
    model = models.mobilenet_v2(pretrained = True)
    
    if 'transfer' in task.split('-'):
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 91)
        optim_params = model.classifier.parameters()
    elif 'tune' in task.split('-'):
        model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 91)
        optim_params = model.parameters()
    else:
        print('Bad Parameter: task must specify what parameters are trainable')
        sys.exit(0)
        
    if initial_model is not None:
        model.load_state_dict(torch.load('./Models/{}'.format(initial_model)))
        
    model.cuda()

    # Setup the optimization process
    metric_loss = torch.nn.BCEWithLogitsLoss()
    
    coco = COCO('{}/annotations/instances_{}{}.json'.format(root, 'val', year))
    cats = coco.loadCats(coco.getCatIds())
    
    def get_counts(y_hat, y):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_hat)):
            if y[i] == y_hat[i] == 1:
               TP += 1
            if y_hat[i] == 1 and y[i] == 0:
               FP += 1
            if y[i] == y_hat[i] == 0:
               TN += 1
            if y_hat [i] == 0 and y[i] == 1:
               FN += 1

        return [TP, FP, TN, FN]
    
    def metric_acc_batch(y_hat, y, cats = cats):
        y_hat = y_hat.cpu().data.numpy()
        y_hat = 1 * (y_hat >= 0)
        y = y.cpu().data.numpy()
        
        out = np.zeros((len(cats), 4))
        c = 0
        for cat in cats:
            index = cat['id']
            out[c, :] = get_counts(np.squeeze(y_hat[:, index]), np.squeeze(y[:, index]))
            c += 1
         
        return out
        
    def metric_acc_agg(counts_list = None):
        if counts_list is None:
            return ['F1', 'Precision', 'Recall']
        else:
            counts_agg = sum(counts_list)
            num_classes = counts_agg.shape[0]
            
            precision = np.zeros((num_classes))
            recall = np.zeros((num_classes))
            f1 = np.zeros((num_classes))
            for i in range(num_classes):
                counts = counts_agg[i, :]
                precision[i] = counts[0] / max((counts[0] + counts[1]), 1)
                recall[i] = counts[0] / max((counts[0] + counts[3]), 1)
                if precision[i] != 0 and recall[i] != 0:
                    f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
                else:
                    f1[i] = 0

            return [np.mean(f1), np.mean(precision), np.mean(recall)]

    name = './Models/{}/model_{}'.format(task, trial)
    model = train_model(model, optim_params, dataloaders, metric_loss, metric_acc_batch, metric_acc_agg, name = name)
    torch.save(model.state_dict(), '{}.pt'.format(name))

if __name__ == '__main__':
    
    worker_id = int(sys.argv[1])
    
    with open('Configs.json', 'r') as f:
        configs = json.load(f)
    configs = configs[worker_id]
    
    for config in configs:
        run(config)
