
import json
from sklearn.model_selection import train_test_split
import sys
import torch
import torch.optim as optim
import torchvision.models as models

from Dataset import unpack_sources, ImageDataset, my_dataloader

sys.path.insert(0, '../Common/')
from Train_new import train_model

def run(config):
    print(config)
    
    root = config[0]
    year = config[1]
    trial = config[2]
    task = config[3]
    
    # Setup the sources for the datasets and the intial model parameters
    if task in ['initial-transfer', 'initial-tune']:
        sources = []
        sources.append('{}/{}{}-info.p'.format(root, 'train', year))
        
        if task == 'initial-tune':
            initial_model = 'initial-transfer/model_{}.pt'.format(trial)
            
    if task in ['random-tune']:
        sources = []
        sources.append('{}/{}{}-info.p'.format(root, mode, year))
        sources.append('{}/{}{}-random-info.p'.format(root, 'train', year))
        
        initial_model = 'initial-tune/model_{}.pt'.format(trial)
            
    if task in ['random-tune-paint']:
        sources = []
        sources.append('{}/{}{}-info.p'.format(root, mode, year))
        sources.append('{}/{}{}-random-paint-info.p'.format(root, 'train', year))
        
        initial_model = 'initial-tune/model_{}.pt'.format(trial)
    
    # Unpack the sources and divide the data into training and validation datasets
    filenames, labels = unpack_sources(sources)
    filenames_train, filenames_val, labels_train, labels_val = train_test_split(filenames, labels, test_size = 0.1)

    if task in ['initial-transfer', 'initial-tune', 'random-tune', 'random-tune-paint']:
        datasets = {}
        datasets['train'] = ImageDataset(filenames_train, labels_train)
        datasets['val'] = ImageDataset(filenames_val, labels_val)
        
        dataloaders = {}
        dataloaders['train'] = my_dataloader(datasets['train'])
        dataloaders['val'] = my_dataloader(datasets['val'])
    
    # Setup the model (initial parameters and what parameters are trainable)
    model = models.mobilenet_v2(pretrained = True)
    
    if task.split('-')[1] == 'transfer':
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 91)
        optim_params = model.classifier.parameters()
        
    elif task.split('-')[1] == 'tune':
        model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 91)
        model.load_state_dict(torch.load('./Models/{}'.format(initial_model)))
        optim_params = model.parameters()
        
    model.cuda()

    # Setup the optimization process
    criterion = torch.nn.BCEWithLogitsLoss()
    lr = 0.001
    optimizer = optim.Adam(optim_params, lr = lr)

    name = './Models/{}/model_{}'.format(task, trial)
    if task == 'initial-transfer':
        model = train_model(model, dataloaders, criterion, optimizer, lr, lr_decay_delay = 2, max_epochs = 10, name = name)
    else:
        model = train_model(model, dataloaders, criterion, optimizer, lr, name = name)
    torch.save(model.state_dict(), '{}.pt'.format(name))

if __name__ == '__main__':
    
    worker_id = int(sys.argv[1])
    
    with open('Configs.json', 'r') as f:
        configs = json.load(f)
    configs = configs[worker_id]
    
    for config in configs:
        run(config)
