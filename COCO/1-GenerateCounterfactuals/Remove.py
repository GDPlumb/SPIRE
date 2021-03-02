
import json
import numpy as np
import os
from pathlib import Path
import sys

sys.path.insert(0, '../')
from Config import get_data_dir, get_random_seed

sys.path.insert(0, '../../Common')
from COCOWrapper import COCOWrapper, id_from_path
from FormatData import mask_images_parallel

if __name__ == '__main__':
    
    # Configuration
    with open('../0-FindPairs/Pairs.json', 'r') as f:
        pairs = json.load(f)
    max_samples = {'train': None, 'val': 1000}
    mask_lists = {'train': ['box'], 'val': []}
    np.random.seed(get_random_seed())
        
    for mode in ['train']:
        mode_dir = '{}/{}'.format(get_data_dir(), mode)
        
        mask_list = mask_lists[mode]
        num_samples = max_samples[mode]
        
        # Setup COCO
        coco = COCOWrapper(mode = mode)
        coco.construct_id2img()
        
        # For each of the identified object pairs    
        for pair in pairs:
            main = pair.split('-')[0]
            spurious = pair.split('-')[1]
            
            pair_dir = '{}/{}-{}'.format(mode_dir, main, spurious)

            # Load the Splits
            with open('{}/{}/splits/{}-{}.json'.format(get_data_dir(), mode, main, spurious), 'r') as f:
                splits = json.load(f)
            
            # Create the counterfactual images
            configs = [('both', main), ('both', spurious), \
                       ('just_main', main), \
                       ('just_spurious', spurious)]
            for config in configs:
                # Get the base images
                name = config[0]            
                imgs = coco.get_imgs_by_ids(splits[name])                          
                if num_samples is not None and num_samples < len(imgs):
                    imgs = np.random.choice(imgs, size = num_samples, replace = False)

                # Get which object is being removed
                chosen_class = config[1]
                chosen_id = coco.get_class_id(chosen_class)
                
                if chosen_class == main:
                    class_type = 'main'
                    unmask = False
                    unmask_classes = None
                elif chosen_class == spurious:
                    class_type = 'spurious'
                    unmask = True
                    unmask_classes = [coco.get_class_id(main)]
                
                config_dir = '{}/{}-{}'.format(pair_dir, name, class_type)
                for mask_mode in mask_list:
                    # Setup the output directory
                    save_dir = '{}/{}'.format(config_dir, mask_mode)
                    os.system('rm -rf {}'.format(save_dir))
                    Path(save_dir).mkdir(parents = True)
                    print(save_dir)
                    
                    # Mask the chosen object
                    filenames, labels = mask_images_parallel(imgs, coco,
                                            save_dir,
                                            chosen_id = chosen_id, mode = mask_mode,
                                            unmask = unmask, unmask_classes = unmask_classes,
                                            use_png = (mask_mode == 'pixel'))

                    # Save the output
                    images = {}
                    for i in range(len(filenames)):
                        filename = filenames[i]
                        label = list(labels[i])
                        id = id_from_path(filename)
                        images[id] = [filename, label]
                    
                    with open('{}/images.json'.format(save_dir), 'w') as f:
                        json.dump(images, f)
