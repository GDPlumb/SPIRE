
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
    with open('../0-FindPairs/Pairs.json', 'r') as f:
        pairs = json.load(f)
    
    for pair in pairs:
        main = pair.split(' ')[0]
        spurious = pair.split(' ')[1]
        pair_dir = '{}/{}-{}'.format(get_data_dir(), main, spurious)
        
        np.random.seed(get_random_seed())
        
        for mode in ['val', 'train']:
            mode_dir = '{}/{}'.format(pair_dir, mode)
            
            # Setup COCO
            coco = COCOWrapper(mode = mode)
            coco.construct_id2img()
            
            # Load the Splits
            with open('{}/splits.json'.format(mode_dir), 'r') as f:
                splits = json.load(f)

            # Create the counterfactual images
            for config in [('just_spurious', main, 'just_main', 1), \
                            ('just_main', spurious, 'just_spurious', 1), \
                            ('neither', main, 'both', 1), \
                            ('neither', spurious, 'both', 0)]:
                
                # Get the 'background' images that will have objects added to them
                name = config[0]
                imgs = coco.get_imgs_by_ids(splits[name])
                                
                # Get the object that will be added
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
                               
                # Get the 'source' images that have that object
                # -  For the training data, we add the object from config[2][i] to the image from config[0][i]
                # -  Becuase config[2] and config[0] have the same size these datasets, this doesn't leak information
                imgs_with_object = coco.get_imgs_by_ids(splits[config[2]]) 
                if mode == 'val':
                    imgs_with_object = np.random.choice(imgs_with_object, size = len(imgs), replace = True)
                
                # Get the label for these counterfactual images
                label = config[3]

                # Setup the output directory
                save_dir = '{}/{}+{}'.format(mode_dir, name, class_type)
                os.system('rm -rf {}'.format(save_dir))
                Path(save_dir).mkdir(parents = True)
                print(save_dir)
                
                # Merge the images
                filenames, _ = add_images_parallel(imgs, imgs_with_object, coco,
                                                        save_dir, 
                                                        chosen_id = chosen_id, mode = 'pixel',
                                                        unmask = unmask, unmask_classes = unmask_classes)
                
                # Save the output
                images = {}
                for filename in filenames:
                    id = id_from_path(filename)
                    images[id] = [filename, label]

                with open('{}/images.json'.format(save_dir), 'w') as f:
                    json.dump(images, f)
