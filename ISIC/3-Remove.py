
import cv2
import joblib
import json
from multiprocessing import Pool
import os
from PIL import Image

from Config import *

def worker(i):
    if i in masks[split]:
        image = np.array(Image.open(dataset[split][i][0]))
        mask = np.array(Image.open(masks[split][i][1])) / 255 # index 1 -> Box            
        image_new = image * (1 - mask) + [124, 116, 104] * mask
        image_new = np.array(image_new, dtype = np.uint8)
        image_new = Image.fromarray(image_new)
        image_new.save('{}/{}.jpg'.format(out_dir, i))
            
if __name__ == '__main__':
    
    out_dir = get_working_dir(mode = 'remove')
    os.system('rm -rf {}'.format(out_dir))
    os.system('mkdir {}'.format(out_dir))
    
    with open('{}/dataset.json'.format(get_working_dir()), 'r') as f:
        dataset = json.load(f)
    
    with open('{}/masks.json'.format(get_working_dir(mode = 'mask')), 'r') as f:
        masks = json.load(f)
    
    for split in dataset:
        p = Pool()
        results = p.map(worker, list(dataset[split]))