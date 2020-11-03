

import os
import pickle
import random
import sys

from Config import get_data_dir
from Misc import get_pair, id_from_path

sys.path.insert(0, '../COCO/')
from COCOWrapper import COCOWrapper
from FormatData import mask_images_parallel

if __name__ == '__main__':

    main = sys.argv[1]
    spurious = sys.argv[2]
    
    # Setup the main directory for this pair
    pair_dir = '{}/{}-{}'.format(get_data_dir(), main, spurious)
    os.system('rm -rf {}'.format(pair_dir))
    os.system('mkdir {}'.format(pair_dir))
    
    # Sample the images that will be used as the base images for these experiments
    num_samples = 1000 #Only applies to train
    for mode in ['val', 'train']:
        mode_dir = '{}/{}'.format(pair_dir, mode)
        os.system('mkdir {}'.format(mode_dir))

        # Get the image splits
        coco = COCOWrapper(mode = mode)
        coco_dir = coco.get_base_dir()
        
        both, just_main, just_spurious, neither = get_pair(coco, main, spurious)
                
        imgs = coco.get_images_with_cats(None)
        id2img = {}
        for img in imgs:
            id2img[id_from_path(img['file_name'])] = img
    
        # Randomly choose a sample from each of the splits to use all experiments
        if mode == 'val':
            splits = {}
            splits['both'] = list(both)
            splits['just_main'] = list(just_main)
            splits['just_spurious'] = list(just_spurious)
            splits['neither'] = list(neither)
        elif mode == 'train':
            splits = {}
            splits['both'] = random.sample(list(both), num_samples)
            splits['just_main'] = random.sample(list(just_main), num_samples)
            splits['just_spurious'] = random.sample(list(just_spurious), num_samples)
            splits['neither'] = random.sample(list(neither), num_samples)
        
        # Create the datastructures to store the images
        # splits:  maps from Split to Image ID
        # images:  maps from Image ID to each available version of that Image (location, label)
        names = ['both', 'just_main', 'just_spurious', 'neither']
        images ={}
        for name in names:
            if name in ['both', 'just_main']:
                label = 1
            else:
                label = 0
              
            tmp = []
            for f in splits[name]:
                id = id_from_path(f)
                tmp.append(id)
                images[id] = {'orig': [f, label]}
            splits[name] = tmp
        
        with open('{}/splits.p'.format(mode_dir), 'wb') as f:
            pickle.dump(splits, f)

        # Create the masked images
        configs = [('both', main, 'both-main', 0), ('both', spurious, 'both-spurious', 1), ('just_main', main, 'just_main-main', 0), ('just_spurious', spurious, 'just_spurious-spurious', 0)]
        for config in configs:
        
            name = config[0]
            ids = splits[name]
            imgs = [id2img[id] for id in ids]

            chosen_class = config[1]
            chosen_id = coco.get_class_id(chosen_class)
            
            if chosen_class == main:
                mod = 'main'
            elif chosen_class == spurious:
                mod = 'spurious'

            config_dir = '{}/{}'.format(mode_dir, config[2])
            os.system('mkdir {}'.format(config_dir))
            
            label = config[3]

            for mask_mode in ['box', 'pixel']:

                mask_dir = '{}/{}'.format(config_dir, mask_mode)
                os.system('mkdir {}'.format(mask_dir))

                mask_images_parallel(imgs, coco.coco, coco_dir, mask_dir, mode = mask_mode, use_png = True, chosen_id = chosen_id)
                
                for id in ids:
                    images[id]['{}-{}'.format(mask_mode, mod)] = ['{}/{}png'.format(mask_dir, id2img[id]['file_name'][:-3]), label]

            os.system('rm {}/*.p'.format(config_dir)) # Remove the autogenerated info file

        # Save
        with open('{}/images.p'.format(mode_dir), 'wb') as f:
            pickle.dump(images, f)
