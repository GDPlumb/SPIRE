
import json
import numpy as np
import os
from pathlib import Path
import pickle
import sys

from Config import get_data_dir, get_random_seed, get_data_fold, get_max_samples

sys.path.insert(0, '../Common')
from COCOHelper import id_from_path
from COCOWrapper import COCOWrapper
from FormatData import mask_images_parallel

if __name__ == '__main__':

    print('Masking')
    
    np.random.seed(get_random_seed())
    
    df = get_data_fold()
    mask_lists = {}
    max_samples = {}
    if df == -1:
        mask_lists['val'] = ['box']
        mask_lists['train'] = ['box', 'pixel']
        
        max_samples['val'] = None
        max_samples['train'] = get_max_samples()
    else:
        mask_lists['val'] = ['box', 'pixel']
        mask_lists['train'] = ['box']
        
        max_samples['val'] = get_max_samples()
        max_samples['train'] = None
    
    with open('./FindSCs.json', 'r') as f:
        pairs = json.load(f)
        
    for mode in ['val', 'train']:
        mode_dir = '{}/{}'.format(get_data_dir(), mode)
        
        mask_list = mask_lists[mode]
        num_samples = max_samples[mode]
            
        coco = COCOWrapper(mode = mode)
        
        imgs = coco.get_images_with_cats(None)
        id2img = {}
        for img in imgs:
            id2img[id_from_path(img['file_name'])] = img
            
        for pair in pairs:
            main = pair.split('-')[0]
            spurious = pair.split('-')[1]
            pair_dir = '{}/{}-{}'.format(mode_dir, main, spurious)
            print(pair_dir)
            
            with open('{}/{}/splits/{}-{}.json'.format(get_data_dir(), mode, main, spurious), 'r') as f:
                splits = json.load(f)
            
            type2name = {}
            type2name['main'] = main
            type2name['spurious'] = spurious
            
            name_base = '{}-{}'.format(main, spurious)
            
            configs = [('both', 'main'), ('both', 'spurious'), ('just_main', 'main'), ('just_spurious', 'spurious')]
            for config in configs:
            
                split_name = config[0]
                split = splits[split_name]
            
                imgs = [id2img[id] for id in split]
                if num_samples is not None and num_samples < len(imgs):
                    imgs = np.random.choice(imgs, size = num_samples, replace = False)

                class_name = config[1]
                chosen_class = type2name[class_name]
                chosen_id = coco.get_class_id(chosen_class)
                
                if class_name == 'main':
                    unmask = False
                    unmask_classes = None
                elif class_name == 'spurious':
                    unmask = True
                    unmask_classes = [coco.get_class_id(main)]

                for mask_mode in mask_list:
                    # Setup the output directory
                    save_dir = '{}/{}-{}-{}'.format(pair_dir, split_name, class_name, mask_mode)
                    Path(save_dir).mkdir(parents = True, exist_ok = True)

                    # Mask Spurious
                    filenames, labels = mask_images_parallel(imgs, coco.coco,
                                            coco.get_base_dir(), save_dir,
                                            chosen_id = chosen_id, mode = mask_mode,
                                            unmask = unmask, unmask_classes = unmask_classes,
                                            use_png = (mask_mode == 'pixel'))

                    # Save the output
                    images = {}
                    for i in range(len(filenames)):
                        filename = filenames[i]
                        label = list(np.copy(labels[i]))
                        id = id_from_path(filename)
                        images[id] = [filename, label]
                    
                    with open('{}/images.json'.format(save_dir), 'w') as f:
                        json.dump(images, f)
