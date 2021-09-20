
'''
How to download ISIC:
-  https://www.isic-archive.com/
-  Click 'Gallery'
-  Click 'Select all on the page for download'
-  Click 'Download as ZIP'
-  Click 'Download all images and metadata in ISIC'
'''

import cv2
import json
import numpy as np
from skimage.segmentation import slic, felzenszwalb, quickshift

def get_data_dir():
    return '/home/gregory/Datasets/ISIC-images'

def get_working_dir(mode = 'real'):
    if mode == 'real':
        return '{}/SP'.format(get_data_dir())
    if mode == 'mask':
        return '{}/SP-masks'.format(get_data_dir())
    if mode == 'remove':
        return '{}/SP-remove'.format(get_data_dir())
    if mode == 'add':
        return '{}/SP-add'.format(get_data_dir())

def get_datasets():
    return ['HAM10000', 'SONIC']

def get_id(filename):
    return filename.split('/')[-1].split('.')[0]

def get_seg(image):
    seg = quickshift(image, ratio = 1.25, kernel_size = 15, sigma = 2.25) #15, 5
    return seg

def get_info(image, seg, min_size = 200):
    out = []
    for i in range(np.max(seg) + 1):
        indicator = 1 * (seg == i)
        count = np.sum(indicator)
        if count >= min_size:
            mask = np.expand_dims(indicator, axis = 2) * image
            scores = np.sum(mask, axis = (0, 1)) / count 
            out.append((scores, indicator, mask))
    return out

class PatchDetector():
    
    def __init__(self, segment_classifier, patch_classes):
        self.segment_classifier = segment_classifier
        self.patch_classes = patch_classes
        
    def predict(self, image):
        seg = get_seg(image)
        info = get_info(image, seg)
        out = {'pred': 0}
        image_cf = np.copy(image)
        for pair in info:
            score = pair[0]
            indicator = pair[1]
            pred = self.segment_classifier.predict(np.expand_dims(score, axis = 0))
            if pred in self.patch_classes:
                out['pred'] = 1
                indices = np.where(indicator == 1)
                left = np.min(indices[1])
                right = np.max(indices[1])
                top = np.min(indices[0])
                bottom = np.max(indices[0])
                cv2.rectangle(image_cf, (left, top), (right, bottom), (255, 0, 0), 2)
                out['cf'] = image_cf
        return out
    
    def get_masks(self, image):
        seg = get_seg(image)
        info = get_info(image, seg)
        out = {'pred': 0}
        mask_pixel = np.zeros(image.size)
        mask_box = np.zeros(image.size)
        for pair in info:
            score = pair[0]
            indicator = pair[1]
            pred = self.segment_classifier.predict(np.expand_dims(score, axis = 0))
            if pred in self.patch_classes:
                out['pred'] = 1
                # Pixel Masking
                mask_pixel += 255 * indicator
                # Box Masking
                indices = np.where(indicator == 1)
                left = np.min(indices[1])
                right = np.max(indices[1])
                top = np.min(indices[0])
                bottom = np.max(indices[0])
                cv2.rectangle(mask_box, (left, top), (right, bottom), (255, 255, 255), -1)
        out['pixel'] = mask_pixel
        out['box'] = mask_box
        return out
    
def get_splits(data_split, source):
    if source == 'true' and data_split == 'test':
        with open('{}/splits.json'.format(get_working_dir()), 'r') as f:
            splits = json.load(f)
            return splits
        
    elif source == 'model' and data_split in ['train', 'test']:
        with open('{}/dataset.json'.format(get_working_dir()), 'r') as f:
            dataset = json.load(f)[data_split]

        with open('{}/masks.json'.format(get_working_dir('mask')), 'r') as f:
            masks = json.load(f)[data_split]
            
        splits = {'both': [], 'just_main': [], 'just_spurious': [], 'neither': []}
        for i in dataset:
            label = dataset[i][1]
            patch = 1 * (i in masks)

            if label == 1:
                if patch == 1:
                    splits['both'].append(i)
                else:
                    splits['just_main'].append(i)
            else:
                if patch == 1:
                    splits['just_spurious'].append(i)
                else:
                    splits['neither'].append(i)

        return splits
    
    print('get_splits(): bad parameters')
    return None