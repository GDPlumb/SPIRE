
import os
from pathlib import Path
import sys
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models

from COCODataset import COCODataset, my_dataloader
from COCOWrapper import COCOWrapper
from ModelWrapper import ModelWrapper
from Train import train_model

n_trials = 5

save_location = './Models/Initial/'

os.system('rm -rf {}'.format(save_location))
Path(save_location).mkdir(parents=True, exist_ok=True)

coco = COCOWrapper()

datasets = {x: COCODataset(mode = x) for x in ['train', 'val']}
dataloaders = {x: my_dataloader(datasets[x]) for x in ['train', 'val']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

for i in range(n_trials):

    # Setup the baseline model for transfer learning
    model = models.mobilenet_v2(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False

    model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 91)
    optim_params = model.classifier.parameters()
    
    model.cuda()
    
    # Run transfer learning
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(optim_params, lr = 0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.1)

    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, num_epochs = 5)
    torch.save(model.state_dict(), '{}/model_{}.pt'.format(save_location, i))
    
    # Evaluate the model on the validation data
    wrapper = ModelWrapper(model)
    y_hat, y_true = wrapper.predict_dataset(dataloaders['val'])
    precision, recall = wrapper.metrics(y_hat, y_true)
    
    original_stdout = sys.stdout

    with open('{}/model_{}.txt'.format(save_location, i), 'w') as f:
        sys.stdout = f
        coco.show_metrics(precision, recall)
        sys.stdout = original_stdout
    
