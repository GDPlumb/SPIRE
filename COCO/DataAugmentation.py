
import numpy as np
import os
from pathlib import Path
import pickle

from COCOWrapper import COCOWrapper

from ImageLoader import ImageLoader



# coco is a COCOWrapper, loader is an ImageLoader
def create(model_class, labeler_classes, coco, loader, mask_mode = 'box', mask_unmask = True, mask_value = 'default', save_dir = './DataAugmentation'):
    
    save_location = '{}/{}{}-{}-{}/{}-{}-{}/'.format(save_dir, coco.mode, coco.year, model_class, labeler_classes, mask_mode, mask_unmask, mask_value).replace("'", '').replace(" ", '')

    os.system('rm -rf {}'.format(save_location))
    Path(save_location).mkdir(parents=True, exist_ok=True)
    
    # Start by getting all of the images that contain the target classes according to the labeler
    relevant_classes = []
    if model_class != 'none':
        relevant_classes.append(model_class)
    for c in labeler_classes:
        relevant_classes.append(c)
    
    img_objs = coco.get_images_with_cats(relevant_classes)
    n = len(img_objs)
    
    labels = []
    labels_masked = coco.get_cat_ids(labeler_classes)
    for i in range(n):
        img_obj = img_objs[i]
        file = '{}{}'.format(save_location, img_obj['file_name'])
        file = '{}png'.format(file[:-3])# Use png for so the pixel values are preserved
     
        # Load and save the masked version of the image
        img = loader.load_img(img_obj, transform_apply = False, mask_apply = True, mask_classes = labeler_classes, mask_mode = mask_mode, mask_unmask = mask_unmask, mask_value = mask_value)
        img.save(file)
        
        # Get the label of the image
        annotations = coco.get_annotations(img_obj)
        label = np.zeros((91), dtype = np.float32)
        for ann in annotations:
            label[ann['category_id']] = 1.0
                
        # Remove the masked objects from the label
        label[labels_masked] = 0.0

        labels.append((file, label))
    
    with open('{}labels.p'.format(save_location), 'wb') as f:
        pickle.dump(labels, f)
        
if __name__ == "__main__":


    root = '/home/gregory/Datasets/COCO/'
    year = '2017'

    model_class = 'none'
    labeler_classes = ['person']
    
    for mode in ['train']: #['val', 'train']:
    
        base = '{}{}{}/'.format(root, mode, year)

        coco = COCOWrapper(root = root, mode = mode, year = year)
        
        loader = ImageLoader(root = base, coco = coco.coco)
        
        for mask_mode in ['box', 'pixel']:
            for mask_unmask in [True]:
                for mask_value in ['default']: #, 'random', 'mean']:
                    create(model_class, labeler_classes, coco, loader, mask_mode = mask_mode, mask_unmask = mask_unmask, mask_value = mask_value)

    for mode in ['val']:
    
        base = '{}{}{}/'.format(root, mode, year)

        coco = COCOWrapper(root = root, mode = mode, year = year)
        
        loader = ImageLoader(root = base, coco = coco.coco)
        
        for mask_mode in ['box', 'pixel']:
            for mask_unmask in [True]:
                for mask_value in ['default', 'random', 'mean']:
                    create(model_class, labeler_classes, coco, loader, mask_mode = mask_mode, mask_unmask = mask_unmask, mask_value = mask_value)
