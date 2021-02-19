
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, '../')
from Config import get_data_dir, get_random_seed

sys.path.insert(0, '../../Common/')
from COCOWrapper import COCOWrapper, id_from_path

if __name__ == '__main__':
    
    # Configuration
    pairs = ['bottle person', 'bowl person', 'car person', 'chair person', 'cup person', 'dining+table person', \
             'bottle cup', 'bowl cup', 'chair cup', \
             'bottle dining+table', 'bowl dining+table', 'chair dining+table', 'cup dining+table']
    num_samples_train = 1000 # We need there to be at least this many images per split.  
    num_samples_val = 500 # We do not need there to be this many images per split.  This is an upper bound to make evaluation quicker.
    
    # Setup
    for pair in  pairs:
        main = pair.split(' ')[0]
        spurious = pair.split(' ')[1]

        pair_dir = '{}/{}-{}'.format(get_data_dir(), main, spurious)
        os.system('rm -rf {}'.format(pair_dir))
        Path(pair_dir).mkdir(parents = True)
        
        random.seed(get_random_seed()) # Setting the random seed per pair makes the results consistent if 'pairs' is changed

        # Sample the images that will be used as the base images for these experiments
        for mode in ['val', 'train']:
            mode_dir = '{}/{}'.format(pair_dir, mode)
            os.system('mkdir {}'.format(mode_dir))

            # Get the image splits
            coco = COCOWrapper(mode = mode)

            splits = coco.get_splits_pair(main, spurious)

            # Randomly choose a sample from each of the splits to use for all experiments
            if mode == 'val':
                for name in splits:
                    tmp = splits[name]
                    tmp = random.sample(list(tmp), min(len(tmp), num_samples_val))
                    splits[name] = tmp
            elif mode == 'train':
                for name in splits:
                    tmp = splits[name]
                    tmp = random.sample(list(tmp), num_samples_train)
                    splits[name] = tmp

            # Create the datastructures to store the images
            images = {}
            for name in splits:
                if name in ['both', 'just_main']:
                    label = 1
                else:
                    label = 0

                tmp = []
                for filename in splits[name]:
                    id = id_from_path(filename)
                    tmp.append(id)
                    images[id] = [filename, label]
                splits[name] = tmp

            with open('{}/images.json'.format(mode_dir), 'w') as f:
                json.dump(images, f)
            
            with open('{}/splits.json'.format(mode_dir), 'w') as f:
                json.dump(splits, f)
            