
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
        
        with open('{}/images.json'.format(mode_dir), 'r') as f:
            images = json.load(f)
            
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
            both = splits['both']
            
            configs = [(both, spurious, '{}-{}'.format(main, spurious)), (both, main, '{}-{}'.format(main, main))]
            for config in configs:
            
                imgs = [id2img[id] for id in config[0]]
                if num_samples is not None and num_samples < len(imgs):
                    imgs = np.random.choice(imgs, size = num_samples, replace = False)

                chosen_class = config[1]
                chosen_id = coco.get_class_id(chosen_class)
                
                if chosen_class == main:
                    unmask = False
                    unmask_classes = None
                elif chosen_class == spurious:
                    unmask = True
                    unmask_classes = [coco.get_class_id(main)]

                name = config[2]

                for mask_mode in mask_list:

                    # Setup the output directory
                    save_dir = '{}/{}-{}'.format(pair_dir, chosen_class, mask_mode)
                    Path(save_dir).mkdir(parents = True, exist_ok = True)

                    # Mask Spurious
                    mask_images_parallel(imgs, coco.coco,
                                        coco.get_base_dir(), save_dir,
                                        chosen_id = chosen_id, mode = mask_mode,
                                        unmask = unmask, unmask_classes = unmask_classes,
                                        use_png = (mask_mode == 'pixel'))

                    # Save the results
                    with open('{}-info.p'.format(save_dir), 'rb') as f:
                        filenames, labels = pickle.load(f)
                    os.system('rm {}-info.p'.format(save_dir))

                    for i in range(len(filenames)):
                        filename = filenames[i]
                        label = list(np.copy(labels[i]))
                        id = id_from_path(filename)

                        images[id]['{}-{}'.format(name, mask_mode)] = [filename, label]

                    with open('{}/images.json'.format(mode_dir), 'w') as f:
                        json.dump(images, f)
