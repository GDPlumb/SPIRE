
import json
import pickle
import sys
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models

from Dataset import ImageDataset, my_dataloader

sys.path.insert(0, '../Common/')
from Train_new import train_model

def run(config):
    print(config)
    
    root = config[0]
    year = config[1]
    trial = config[2]
    task = config[3]
        
    if task in ['initial-transfer', 'initial-tune']:
        sources = {}
        for mode in ['train', 'val']:
            sources[mode] = []
            
            sources[mode].append('{}/{}{}-info.p'.format(root, mode, year))
            
    if task in ['random-transfer', 'random-tune']:
        sources = {}
        for mode in ['train', 'val']:
            sources[mode] = []
            
            sources[mode].append('{}/{}{}-info.p'.format(root, mode, year))
            sources[mode].append('{}/{}{}-random-info.p'.format(root, mode, year))
            
    if task in ['random-tune-paint']:
        sources = {}
        for mode in ['train', 'val']:
            sources[mode] = []
            
            sources[mode].append('{}/{}{}-info.p'.format(root, mode, year))
            sources[mode].append('{}/{}{}-random-paint-info.p'.format(root, mode, year))
            
    if task == 'augment-transfer':
        sources = {}
        for mode in ['train', 'val']:
            sources[mode] = []
            
            spurious_class = config[4]
            
            sources[mode].append('{}/{}{}-info.p'.format(root, mode, year))
            sources[mode].append('{}/{}{}-{}-info.p'.format(root, mode, year, spurious_class))
            
    if task == 'both-transfer':
        sources = {}
        for mode in ['train', 'val']:
            sources[mode] = []
            
            spurious_class = config[4]
            
            sources[mode].append('{}/{}{}-info.p'.format(root, mode, year))
            sources[mode].append('{}/{}{}-random-info.p'.format(root, mode, year))
            sources[mode].append('{}/{}{}-{}-info.p'.format(root, mode, year, spurious_class))

        
    if task in ['initial-transfer', 'random-transfer', 'augment-transfer', 'both-transfer', 'initial-tune', 'random-tune', 'random-tune-paint']:
    
        datasets = {}
        dataloaders = {}
        for mode in ['train', 'val']:
            datasets[mode] = ImageDataset(sources[mode])
            dataloaders[mode] = my_dataloader(datasets[mode])
            
            
    model = models.mobilenet_v2(pretrained = True)
    
    if task.split('-')[1] == 'transfer':
        for param in model.parameters():
            param.requires_grad = False

        model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 91)
        optim_params = model.classifier.parameters()
    elif task.split('-')[1] == 'tune':
        model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 91)
        model.load_state_dict(torch.load('./Models/initial-transfer/model_0.pt'))
        optim_params = model.parameters()
        
    model.cuda()

    criterion = torch.nn.BCEWithLogitsLoss()
    lr = 0.001
    optimizer = optim.Adam(optim_params, lr = lr)

    name = './Models/{}/model_{}'.format(task, trial)
    model = train_model(model, dataloaders, criterion, optimizer, lr, name = name)
    torch.save(model.state_dict(), '{}.pt'.format(name))

if __name__ == '__main__':
    
    worker_id = int(sys.argv[1])
    
    with open('Configs.json', 'r') as f:
        configs = json.load(f)
    configs = configs[worker_id]
    
    for config in configs:
        run(config)
