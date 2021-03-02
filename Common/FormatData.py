
from multiprocessing.pool import ThreadPool
import numpy as np
import os
from PIL import Image
from pycocotools.coco import COCO
import random
import sys
from torchvision import transforms

from COCOWrapper import id_from_path
from Dataset import MakeSquare

def get_mask(anns, mask_classes, coco, mode = 'box', unmask = True, unmask_classes = None):

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
            
            if unmask_classes is None:
                def check(cat):
                    return (cat not in mask_classes)
            else:
                def check(cat):
                    return (cat in unmask_classes)
            
            for ann in anns:
                if check(ann['category_id']):
                    tmp = coco.annToMask(ann)
                    unmask.append(tmp)
            if len(unmask) > 0:
                unmask = np.expand_dims(1.0 * (np.sum(np.array(unmask), axis = 0) >= 1.0), axis = 2)
                mask = mask - unmask
                mask = np.clip(mask, 0, 1)
                
        return mask
    else:
        return None
        
def apply_mask(img, mask, value = 'default', invert = False):
    # Fill the masked values
    if invert:
        mask = (np.squeeze(mask) != 1)
    else:
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

def mask_images(images, coco, 
                save_location, 
                chosen_id = None, mode = 'box', invert = False, 
                unmask = True, unmask_classes = None, 
                use_png = False):
    
    base_location = coco.get_base_dir()
    coco = coco.coco
    
    filenames = []
    labels = []
    for img_obj in images:
    
        base_filename = img_obj['file_name']
        img = Image.open('{}/{}'.format(base_location, base_filename)).convert('RGB')
        anns = coco.loadAnns(coco.getAnnIds(imgIds = img_obj['id']))
        
        filename = '{}/{}'.format(save_location, base_filename)
        label = np.zeros((91))
        for ann in anns:
            label[ann['category_id']] = 1.0
        
        if len(anns) > 0:
        
            # Randomly select one of the object categories present to mask
            if chosen_id is None:
                tmp_id = random.choice(list(set([ann['category_id'] for ann in anns])))
            else:
                tmp_id = chosen_id
                
            # Modify the label
            label[tmp_id] = 0.0
            
            # Mask the object
            if not isinstance(tmp_id, list):
                tmp_id = [tmp_id]
            mask = get_mask(anns, tmp_id, coco, mode = mode, unmask = unmask, unmask_classes = unmask_classes)
            img = apply_mask(img, mask, invert = invert)

        # Save the output
        if use_png: # Preserves exact pixel values - used to pass the masked pixels to the inpainter
            filename = '{}png'.format(filename[:-3])
        filenames.append(filename)
        labels.append(label)
        img.save(filename)
        
    return filenames, labels
    
def mask_images_parallel(images, coco, 
                         save_location, 
                         chosen_id = None, mode = 'box', invert = False, 
                         unmask = True, unmask_classes = None, 
                         use_png = False, 
                         workers = 24):

    # Split the images to pass them to the workers
    images_split = []
    for i in range(workers):
        images_split.append([])
    
    next_worker = 0
    for image in images:
        images_split[next_worker].append(image)
        next_worker = (next_worker + 1) % workers
        
    # Define the worker function
    def mask_images_worker(id, images_split = images_split, coco = coco, 
                           save_location = save_location, 
                           chosen_id = chosen_id, mode = mode, invert = invert, 
                           unmask = unmask, unmask_classes = unmask_classes, 
                           use_png = use_png):
        return mask_images(images_split[id], coco,
                           save_location, 
                           chosen_id = chosen_id, mode = mode, invert = invert,
                           unmask = unmask, unmask_classes = unmask_classes, 
                           use_png = use_png)
    
    # Run
    pool = ThreadPool(processes = workers)
    out = pool.map(mask_images_worker, range(workers))
    
    # Collect the output
    filenames = []
    labels = []
    for pair in out:
        f = pair[0]
        l = pair[1]
        for i in range(len(f)):
            filenames.append(f[i])
            labels.append(l[i])
    return filenames, labels

def get_custom_resize(d):
    return transforms.Compose([
            MakeSquare(),
            transforms.Resize((d,d))
            ])

def add_images(images, images_with_object, coco,
               save_location, 
               chosen_id = None, mode = 'pixel',
               unmask = False, unmask_classes = None):
        
    base_dir = coco.get_base_dir()
    coco = coco.coco
    
    filenames = []
    labels = []
    for i, img_obj in enumerate(images):
                
        base_filename = img_obj['file_name']
        img = np.array(Image.open('{}/{}'.format(base_dir, base_filename)).convert('RGB'))
        width, height, _ = img.shape
        dim_min = min(width, height)
        anns = coco.loadAnns(coco.getAnnIds(imgIds = img_obj['id']))

        img_object_obj = images_with_object[i]
        img_object = Image.open('{}/{}'.format(base_dir, img_object_obj['file_name'])).convert('RGB')
        anns_object = coco.loadAnns(coco.getAnnIds(imgIds = img_object_obj['id']))    
        
        filename = '{}/{}'.format(save_location, base_filename)
        label = np.zeros((91))
        for ann in anns:
            label[ann['category_id']] = 1.0
        
        if len(anns_object) > 0:
            
            # Randomly select one of the object categories present to add
            if chosen_id is None:
                tmp_id = random.choice(list(set([ann['category_id'] for ann in anns_obj])))
            else:
                tmp_id = chosen_id
            
            # Modify the label
            label[tmp_id] = 1.0
            
            # Add the object
            if not isinstance(tmp_id, list):
                tmp_id = [tmp_id]
            mask = get_mask(anns_object, tmp_id, coco, mode = mode)
        
            custom_resize = get_custom_resize(dim_min)

            mask = np.array(custom_resize(Image.fromarray(np.squeeze(mask))))
            img_object = np.array(custom_resize(img_object))
            
            if unmask:
                mask_unmask = get_mask(anns, unmask_classes, coco, mode = mode)
                if mask_unmask is not None:
                    mask_unmask = np.squeeze(mask_unmask)
                    mask -= mask_unmask[:dim_min, :dim_min]
                    mask = np.clip(mask, 0, 1)

            mask_indices = np.where(mask != 0) # Mask is no longer binary because we resize
            for j in range(3):
                img[mask_indices[0], mask_indices[1], j] = img_object[mask_indices[0], mask_indices[1], j]

            img = Image.fromarray(np.uint8(img))
        
        # Save the output
        filenames.append(filename)
        labels.append(label)
        img.save(filename)
    
    return filenames, labels

def add_images_parallel(images, images_with_object, coco,
                        save_location, 
                        chosen_id = None, mode = 'pixel',
                        unmask = False, unmask_classes = None,
                        workers = 24):
    
    # Split the images to pass them to the workers
    images_split = []
    images_object_split = []
    for i in range(workers):
        images_split.append([])
        images_object_split.append([])
    
    next_worker = 0
    for i in range(len(images)):
        images_split[next_worker].append(images[i])
        images_object_split[next_worker].append(images_with_object[i])
        next_worker = (next_worker + 1) % workers
        
    # Define the worker function
    def add_images_worker(id, images_split = images_split, images_object_split = images_object_split, coco = coco, 
                          save_location = save_location,
                          chosen_id = chosen_id, mode = mode, 
                          unmask = unmask, unmask_classes = unmask_classes):
        return add_images(images_split[id], images_object_split[id], coco,
                          save_location,
                          chosen_id = chosen_id, mode = mode, 
                          unmask = unmask, unmask_classes = unmask_classes)
        
    # Run
    pool = ThreadPool(processes = workers)
    out = pool.map(add_images_worker, range(workers))
    
    # Collect the output
    filenames = []
    labels = []
    for pair in out:
        f = pair[0]
        l = pair[1]
        for i in range(len(f)):
            filenames.append(f[i])
            labels.append(l[i])
    return filenames, labels
