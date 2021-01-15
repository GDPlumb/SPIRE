
import json
import numpy as np
import os
import sys

from Config import get_data_dir

sys.path.insert(0, '../Common')
from COCOHelper import id_from_path
from COCOWrapper import COCOWrapper
        
if __name__ == '__main__':

    print('Initial Setup')
    
    base_dir = get_data_dir()
    os.system('rm -rf {}'.format(base_dir))
    os.system('mkdir {}'.format(base_dir))

    for mode in ['val', 'train']:
        mode_dir = '{}/{}'.format(base_dir, mode)
        os.system('mkdir {}'.format(mode_dir))
        print(mode_dir)
        
        # Get the images
        coco = COCOWrapper(mode = mode)
        coco_dir = coco.get_base_dir()
                
        imgs = coco.get_images_with_cats(None)
        
        # images:  maps from Image ID to each available version of that Image (location, label)
        images = {}
        for img in imgs:
        
            id = id_from_path(img['file_name'])
            
            filename = '{}/{}'.format(coco_dir, img['file_name'])
            
            anns = coco.coco.loadAnns(coco.coco.getAnnIds(img['id'], iscrowd = None))
            label = np.zeros((91))  # Each 'label' vector is large enough for easy indexing, but this means it contains unused indices
            for ann in anns:
                label[ann['category_id']] = 1.0
            label = list(label)
            
            images[id] = [filename, label]
        
        # Save
        with open('{}/images.json'.format(mode_dir), 'w') as f:
            json.dump(images, f)
            
        # Create a map from object pairs to splits of the dataset
        names = []
        for cat in coco.cats:
            names.append(cat['name'].replace(' ', '+'))
            
        ids_all = [id_from_path(img['file_name']) for img in coco.get_images_with_cats(None)]
           
        name2id = {}
        for name in names:
            name2id[name] = [id_from_path(img['file_name']) for img in coco.get_images_with_cats([name.replace('+', ' ')])]
            
        name2id_not = {}
        for name in names:
            name2id_not[name] = np.setdiff1d(ids_all, name2id[name])
         
        splits_dir = '{}/splits'.format(mode_dir)
        os.system('mkdir {}'.format(splits_dir))
        for main in names:
            ids_main = name2id[main]
            ids_main_not = name2id_not[main]
                    
            for spurious in names:
                if main != spurious:
                    ids_spurious = name2id[spurious]
                    ids_spurious_not = name2id_not[spurious]
                    
                    splits = {}
                    splits['both'] = list(np.intersect1d(ids_main, ids_spurious))
                    splits['just_main'] = list(np.setdiff1d(ids_main, ids_spurious))
                    splits['just_spurious'] = list(np.setdiff1d(ids_spurious, ids_main))
                    splits['neither'] = list(np.intersect1d(ids_main_not, ids_spurious_not))
                    
                    with open('{}/{}-{}.json'.format(splits_dir, main, spurious), 'w') as f:
                        json.dump(splits, f)
