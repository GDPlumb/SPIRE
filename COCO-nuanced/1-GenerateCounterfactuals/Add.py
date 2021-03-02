
import json
import numpy as np
import os
from pathlib import Path
import sys

sys.path.insert(0, '../')
from Config import get_data_dir, get_random_seed

sys.path.insert(0, '../../Common/')
from COCOWrapper import COCOWrapper, id_from_path
from FormatData import add_images_parallel

if __name__ == '__main__':

    # Configuration
    label1 = 'runway'
    label2 = 'street'
    spurious = 'airplane'
    np.random.seed(get_random_seed())
    
    tuple_dir = '{}/{}-{}/{}'.format(get_data_dir(), label1, label2, spurious)
    
    for mode in ['val', 'train']:
        mode_dir = '{}/{}'.format(tuple_dir, mode)

        # Setup COCO
        coco = COCOWrapper(mode = mode)
        coco.construct_id2img()
            
        # Load the Splits
        with open('{}/splits.json'.format(mode_dir), 'r') as f:
            splits = json.load(f)
                    
        splits['spurious'] = []
        for split in ['1s', '0s']:
            for id in splits[split]:
                splits['spurious'].append(id)

        # Create the counterfactual images
        configs = [('1ns', 1), ('0ns', 0)]
        
        chosen_class = spurious
        chosen_id = coco.get_class_id(chosen_class)
        
        class_type = 'spurious'
        unmask = False
        unmask_classes = None
        
        for config in configs:
            
            # Get the 'background' images that will have objects added to them
            name = config[0]
            imgs = coco.get_imgs_by_ids(splits[name])
            
            # Get the 'source' images that have that object
            imgs_with_object = coco.get_imgs_by_ids(splits[class_type])
            imgs_with_object = np.random.choice(imgs_with_object, size = len(imgs), replace = True)
            
            # Get the label for these counterfactual images
            label = config[1]
            
            # Setup the output directory
            save_dir = '{}/{}+{}'.format(mode_dir, name, class_type)
            os.system('rm -rf {}'.format(save_dir))
            Path(save_dir).mkdir(parents = True)
            print(save_dir)
            
            # Merge the images
            filenames, labels = add_images_parallel(imgs, imgs_with_object, coco,
                                                    save_dir, 
                                                    chosen_id = chosen_id, mode = 'pixel',
                                                    unmask = unmask, unmask_classes = unmask_classes)

            # Save the output
            images = {}
            for i in range(len(filenames)):
                filename = filenames[i]
                label = list(labels[i])
                id = id_from_path(filename)
                images[id] = [filename, label]

            with open('{}/images.json'.format(save_dir), 'w') as f:
                json.dump(images, f)
            