
from multiprocessing.pool import ThreadPool
import numpy as np
import os
import pickle
from PIL import Image
from pycocotools.coco import COCO
import random
import sys

def format_standard(root, mode, year):

    coco = COCO('{}/annotations/instances_{}{}.json'.format(root, mode, year))
    images = coco.loadImgs(coco.getImgIds())

    filenames = []
    labels = []
    for img_obj in images:
        filename = '{}/{}{}/{}'.format(root, mode, year, img_obj['file_name'])

        anns = coco.loadAnns(coco.getAnnIds(img_obj['id'], iscrowd = None))
        label = np.zeros((91), dtype = np.float32)  # Each 'label' vector is large enough for easy indexing, but this means it contains unused indices
        for ann in anns:
            label[ann['category_id']] = 1.0

        filenames.append(filename)
        labels.append(label)
        
    with open('{}/{}{}-info.p'.format(root, mode, year), 'wb') as f:
        pickle.dump([filenames, labels], f)

def get_mask(anns, mask_classes, coco, mode = 'box', unmask = True):

    # Find the mask for each object we want to remove
    mask = []
    for ann in anns:
        if ann['category_id'] in mask_classes:
            # Get the pixelwise mask from COCO
            tmp = coco.annToMask(ann)
            # Process that mask
            if mode == 'pixel':
                mask.append(tmp)
            elif mode == 'box':
                idx = np.where(tmp == 1.0)
                if len(idx[0] > 0):
                    min_0 = np.min(idx[0])
                    max_0 = np.max(idx[0])
                    min_1 = np.min(idx[1])
                    max_1 = np.max(idx[1])
                    tmp_new = np.copy(tmp)
                    tmp_new[min_0:max_0, min_1:max_1] = 1.0
                    mask.append(tmp_new)
                else: # BUG?  This handles the images that have some blank annotations
                    mask.append(tmp)

    # If we found at least one object to mask from the image
    if len(mask) > 0:
        mask = np.expand_dims(1.0 * (np.sum(np.array(mask), axis = 0) >= 1.0), axis = 2)
        
        # If we want to unmask any objects other than those we explicitly masked
        if unmask:
            unmask = []
            for ann in anns:
                if ann['category_id'] not in mask_classes:
                    tmp = coco.annToMask(ann)
                    unmask.append(tmp)
            if len(unmask) > 0:
                unmask = np.expand_dims(1.0 * (np.sum(np.array(unmask), axis = 0) >= 1.0), axis = 2)
                mask = mask - unmask
                mask = np.clip(mask, 0, 1)
                
        return mask
        
def apply_mask(img, mask, value = 'default'):
    # Fill the masked values
    mask = (np.squeeze(mask) == 1)
    img_np = np.array(img)
    if value == 'default':
        img_np[mask] = [124, 116, 104]
    elif value == 'random':
        img_np[mask] =  np.random.randint(low = 0, high = 256, size = (np.sum(mask), 3))
    elif value == 'mean':
        img_np[mask] = np.mean(np.array(img), axis = (0,1)).astype(np.int)
    img = Image.fromarray(img_np)
    return img

def mask_images(images, coco, base_location, save_location, chosen_id = None, mode = 'box', unmask = True, use_png = False):
    filenames = []
    labels = []
    for img_obj in images:
    
        base_filename = img_obj['file_name']
        img = Image.open('{}/{}'.format(base_location, base_filename)).convert('RGB')
        
        filename = '{}/{}'.format(save_location, base_filename)
        label = np.zeros((91), dtype = np.float32)

        anns = coco.loadAnns(coco.getAnnIds(imgIds = img_obj['id']))
        
        if len(anns) > 0:
        
            # Randomly select one of the object categories present in this image to mask
            if chosen_id is None:
                tmp_id = random.choice(list(set([ann['category_id'] for ann in anns])))
            else:
                tmp_id = chosen_id
                
            # Setup the label
            for ann in anns:
                label[ann['category_id']] = 1.0
            label[tmp_id] = 0.0
            
            # Mask the image
            mask = get_mask(anns, [tmp_id], coco, mode = mode, unmask = unmask)
            img = apply_mask(img, mask)

        # Save the output
        if use_png: # Preserves exact pixel values - used to pass the masked pixels to the inpainter
            filename = '{}png'.format(filename[:-3])
        filenames.append(filename)
        labels.append(label)
        img.save(filename)
        
    return filenames, labels
    
def mask_images_parallel(images, coco, base_location, save_location, chosen_id = None, mode = 'box', unmask = True, use_png = False, workers = 24):

    # Split the images to pass them to the workers
    images_split = []
    for i in range(workers):
        images_split.append([])
    
    next_worker = 0
    for image in images:
        images_split[next_worker].append(image)
        next_worker = (next_worker + 1) % workers
        
    # Define the worker function
    def mask_images_worker(id, images_split = images_split, coco = coco, base_location = base_location, save_location = save_location, chosen_id = chosen_id, mode = mode, unmask = unmask, use_png = use_png):
        names, labels = mask_images(images_split[id], coco, base_location, save_location, chosen_id = chosen_id, mode = mode, unmask = unmask, use_png = use_png)
        with open('tmp-{}.p'.format(id), 'wb') as f:
            pickle.dump([names, labels], f)
        
    # Run
    pool = ThreadPool(processes = workers)
    pool.map(mask_images_worker, range(workers))
    
    # Collect the output
    filenames = []
    labels = []
    for i in range(workers):
        with open('tmp-{}.p'.format(i), 'rb') as f:
            data = pickle.load(f)
            
        for j in range(len(data[0])):
            filenames.append(data[0][j])
            labels.append(data[1][j])
    
    # Save the output
    with open('{}-info.p'.format(save_location), 'wb') as f:
        pickle.dump([filenames, labels], f)
        
    # Clean up
    os.system('rm tmp-*.p')

def format_random(root, mode, year, mask_mode = 'box', unmask = True, use_png = False):

    # Prep the data directory
    base_location = '{}/{}{}'.format(root, mode, year)
    save_location = '{}-random'.format(base_location)
    if mask_mode == 'pixel':
        save_location = '{}-pixel'.format(save_location)
        
    os.system('rm -rf {}'.format(save_location))
    os.system('mkdir {}'.format(save_location))
    
    # Create a copy of the data where each image has a random object category masked
    coco = COCO('{}/annotations/instances_{}{}.json'.format(root, mode, year))
    images = coco.loadImgs(coco.getImgIds())

    # Run
    mask_images_parallel(images, coco, base_location, save_location, chosen_id = None, mode = mask_mode, unmask = unmask, use_png = use_png)

def format_spurious(root, mode, year, spurious, mask_mode = 'box', unmask = True, use_png = False, coco = None, use_tmp = False):

    # Prep the data directory
    base_location = '{}/{}{}'.format(root, mode, year)
    if use_tmp:
        save_location = '{}/tmp'.format(root)
    else:
        save_location = '{}-{}'.format(base_location, spurious)
        if mask_mode == 'pixel':
            save_location = '{}-pixel'.format(save_location)
    os.system('rm -rf {}'.format(save_location))
    os.system('mkdir {}'.format(save_location))
    
    # Create a copy of the data where each image has a random object category masked
    if coco is None:
        coco = COCO('{}/annotations/instances_{}{}.json'.format(root, mode, year))
    spurious_id = coco.getCatIds(catNms = [spurious])[0]
    images = coco.loadImgs(coco.getImgIds(catIds = spurious_id))
    
    # Run
    mask_images_parallel(images, coco, base_location, save_location, chosen_id = spurious_id, mode = mask_mode, unmask = unmask, use_png = use_png)

if __name__ == '__main__':

    print()
    print(sys.argv)
    print()
    
    root = sys.argv[1]
    mode = sys.argv[2]
    year = sys.argv[3]
    step = sys.argv[4]

    if step == 'standard':
        format_standard(root, mode, year)
    elif step == 'random':
        format_random(root, mode, year)
    elif step == 'random-pixel':
        format_random(root, mode, year, mask_mode = 'pixel', use_png = True)
    elif step == 'spurious':
        spurious = sys.argv[5]
        format_spurious(root, mode, year, spurious)
    elif step == 'spurious-pixel':
        spurious = sys.argv[5]
        format_spurious(root, mode, year, spurious, mask_mode = 'pixel', use_png = True)
