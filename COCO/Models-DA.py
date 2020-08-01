
import os
from pathlib import Path
import sys
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models

from COCODataset import COCODataset, my_dataloader
from Train import train_model

n_trials = 5
save_base= './Models/DA/'

model_class = 'none'
labeler_classes = ['person']
altered_setup = 'box-True-paint'

save_extension = '{}-{}/{}'.format(model_class, labeler_classes, altered_setup).replace("'", '').replace(' ', '')

save_location = '{}{}/'.format(save_base, save_extension)

os.system('rm -rf {}'.format(save_location))
Path(save_location).mkdir(parents=True, exist_ok=True)

sources = {}
for mode in ['val', 'train']:
    sources[mode] = './DataAugmentation/{}2017-{}/'.format(mode, save_extension)

datasets = {x: COCODataset(mode = x, sources = ['{}labels.p'.format(sources[x])]) for x in ['train', 'val']}
dataloaders = {x: my_dataloader(datasets[x]) for x in ['train', 'val']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

for i in range(n_trials):

    # Setup the baseline model for transfer learning
    model = models.mobilenet_v2(pretrained = True)
        
    for param in model.parameters():
        param.requires_grad = False

    model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 91)
    optim_params = model.classifier.parameters()
    
    model.load_state_dict(torch.load('./Models/Initial/model_{}.pt'.format(i))) #Load one of the Initial transfer learned models
    
    model.cuda()
    
    # Run transfer learning
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(optim_params, lr = 0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.1)

    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, num_epochs = 3)
    torch.save(model.state_dict(), '{}/model_{}.pt'.format(save_location, i))
    
    
