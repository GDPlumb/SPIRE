import glob
import json
import numpy as np
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import sys
from torchvision import transforms
from tqdm import tqdm

from Config import *

sys.path.insert(0, '../Common/')
from Dataset import MakeSquare

if __name__ == '__main__':
    
    data_dir = get_data_dir()
    working_dir = get_working_dir()
    os.system('rm -rf {}'.format(working_dir))
    os.system('mkdir {}'.format(working_dir))
    
    def get_resize():
        return transforms.Compose([MakeSquare(), transforms.Resize((224, 224))])
    t = get_resize()

    def resize_image(filename):
        return t(Image.open(filename).convert('RGB'))
    
    # Get the labels (benign vs malignant) for all of the images in ISIC
    meta = pd.read_csv('{}/metadata.csv'.format(data_dir), low_memory = False)
    meta = np.array(meta)

    labels = {}
    for i in range(meta.shape[0]):
        labels[meta[i, 1]] = meta[i, 4]

    # Get the filenames and ids for the images in the chosen ISIC datasets
    filenames = {}
    for dataset in get_datasets(): 
        for filename in glob.glob('{}/{}/*.jpg'.format(data_dir, dataset))  :
            filenames[get_id(filename)] = filename

    # Resize the images 
    # - This will make the segmentation algorithm and model training faster
    # - It may also make the segmentation algorithm work more consistently
    for i in tqdm(filenames):
        im = resize_image(filenames[i])
        im.save('{}/{}.jpg'.format(working_dir, i))
    
    # Create a train-test split
    ids = {}
    ids['train'], ids['test'] = train_test_split(list(filenames), test_size = 0.1)

    dataset = {}
    for mode in ids:
        dataset_tmp = {}
        for i in ids[mode]:
            # Get the label for this image
            v = labels[i]
            if v == 'malignant':
                v = 1
            elif v == 'benign':
                v = 0
            else:
                v = -1
            # If we have a good label, add this image to the dataset
            if v != -1:
                dataset_tmp[i] = ['{}/{}.jpg'.format(working_dir, i), v]
        dataset[mode] = dataset_tmp

    with open('{}/dataset.json'.format(working_dir), 'w') as f:
        json.dump(dataset, f)