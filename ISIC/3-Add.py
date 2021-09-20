
import cv2
import joblib
import json
from multiprocessing import Pool
import os
from PIL import Image
import random

from Config import *

def worker(i):
    if i not in masks[split]:
        image = np.array(Image.open(dataset[split][i][0]))
        i_prime = random.choice(list(masks[split]))
        mask = np.array(Image.open(masks[split][i_prime][0])) / 255 # Pixel
        image_prime = np.array(Image.open(dataset[split][i_prime][0]))
        image_new = image * (1 - mask) + image_prime * mask
        image_new = np.array(image_new, dtype = np.uint8)
        image_new = Image.fromarray(image_new)
        image_new.save('{}/{}.jpg'.format(out_dir, i))

if __name__ == '__main__':
    
    out_dir = get_working_dir(mode = 'add')
    os.system('rm -rf {}'.format(out_dir))
    os.system('mkdir {}'.format(out_dir))
    
    with open('{}/dataset.json'.format(get_working_dir()), 'r') as f:
        dataset = json.load(f)
    
    with open('{}/masks.json'.format(get_working_dir(mode = 'mask')), 'r') as f:
        masks = json.load(f)
        
    for split in dataset:
        p = Pool()
        results = p.map(worker, list(dataset[split]))