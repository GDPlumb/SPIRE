
import glob
import os
import pickle
import sys
import torch
import torchvision.models as models

from Dataset import ImageDataset, my_dataloader

from ModelWrapper import ModelWrapper


if __name__ == '__main__':

    root = sys.argv[1]
    year = sys.argv[2]
    spurious_class = sys.argv[3]
    task = sys.argv[4]
    
    # Setup Datafiles
    datafiles = []
    datafiles.append('{}/val{}-info.p'.format(root, year))
    datafiles.append('{}/train{}-with-{}-info.p'.format(root, year, spurious_class))
    datafiles.append('{}/train{}-without-{}-info.p'.format(root, year, spurious_class))
    datafiles.append('{}/val{}-{}-info.p'.format(root, year, spurious_class))
    
    # This prevents some weird crash
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Setup the model and wrapper
    model = models.mobilenet_v2(pretrained = True)
    model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 91)
    model.eval()
    model.cuda()
    wrapper = ModelWrapper(model)

    for file in glob.glob('./Models/{}/*.pt'.format(task)):
        
        results = {}
        
        model.load_state_dict(torch.load(file))
        
        for datafile in datafiles:
        
            dataset = ImageDataset([datafile])
            dataloader = my_dataloader(dataset)
            
            y_hat, y_true = wrapper.predict_dataset(dataloader)
            precision, recall = wrapper.metrics(y_hat, y_true)
            
            results[datafile] = [precision, recall]
    
        with open('{}.p'.format(file[:-3]), 'wb') as f:
            pickle.dump(results, f)
