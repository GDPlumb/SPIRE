
import glob
import numpy as np
import os
import pickle
import sys
import torch
import torchvision.models as models

from Dataset import ImageDataset, my_dataloader
from ModelWrapper import ModelWrapper

if __name__ == '__main__':

    print(sys.argv)

    task = sys.argv[1]
    
    base_location = '/home/gregory/Datasets/OpenImages'
    
    # Setup Dataset
    filenames = []
    labels = []
    for dataset in ['validation', 'test']:
        for file in glob.glob('{}/{}/*.jpg'.format(base_location, dataset)):
            filenames.append(file)
            labels.append(np.array([0]))
            
    dataset = ImageDataset(filenames, labels, get_names = True)
    dataloader = my_dataloader(dataset)

    # Setup the model and wrapper
    model = models.mobilenet_v2(pretrained = True)
    model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 91)
    model.eval()
    model.cuda()
    wrapper = ModelWrapper(model, get_names = True)

    for file in glob.glob('./Models/{}/*.pt'.format(task)):
        print('Model: ', file)
        model.load_state_dict(torch.load(file))

        y_hat, y_true, names = wrapper.predict_dataset(dataloader)
        
        
        results = {}
        for i in range(len(names)):
            results[names[i].split('/')[-1].split('.')[0]] = y_hat[i, :]
                    
        with open('{}_OpenImages.p'.format(file[:-3]), 'wb') as f:
            pickle.dump(results, f)
