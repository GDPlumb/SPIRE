
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
                
            splits['main'] = []
            for split in ['both', 'just_main']:
                for id in splits[split]:
                    splits['main'].append(id)
                    
            splits['spurious'] = []
            for split in ['both', 'just_spurious']:
                for id in splits[split]:
                    splits['spurious'].append(id)
            
            # Create the counterfactual images
            configs = [('just_main', spurious), \
                       ('just_spurious', main), \
                       ('neither', main), ('neither', spurious)]
            for config in configs:
        
                # Get the 'background' images that will have objects added to them
                name = config[0]
                imgs = coco.get_imgs_by_ids(splits[name])
                if num_samples is not None and num_samples < len(imgs):
                    imgs = np.random.choice(imgs, size = num_samples, replace = False)
                
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
                imgs_with_object = coco.get_imgs_by_ids(splits[class_type])
                imgs_with_object = np.random.choice(imgs_with_object, size = len(imgs), replace = True)
                
                # Setup the output directory
                save_dir = '{}/{}+{}'.format(pair_dir, name, class_type)
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
                
