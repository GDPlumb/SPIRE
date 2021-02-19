
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, '../')
from Config import get_data_dir

sys.path.insert(0, '../../Common')
from COCOWrapper import COCOWrapper, id_from_path

if __name__ == '__main__':
    
    # Configuration
    label1 = 'runway'
    label2 = 'street'
    spurious = 'airplane'
    
    # Setup
    tuple_dir = '{}/{}-{}/{}'.format(get_data_dir(), label1, label2, spurious)
    os.system('rm -rf {}'.format(tuple_dir))
    Path(tuple_dir).mkdir(parents = True)
    
    for mode in ['val', 'train']:
        mode_dir = '{}/{}'.format(tuple_dir, mode)
        os.system('mkdir {}'.format(mode_dir))
        
        # Get the image splits
        coco = COCOWrapper(mode = mode)
        coco.setup_caption_maps()
        
        splits = coco.get_splits_words(label1, label2, spurious)
        with open('{}/splits.json'.format(mode_dir), 'w') as f:
            json.dump(splits, f)
    
        # Create the datastructures to store the images
        images = {}
        for name in splits:
            if name in ['1s', '1ns']:
                label = 1
            else:
                label = 0
              
            for filename in splits[name]:
                id = id_from_path(filename)
                images[id] = [filename, label]
        
        with open('{}/images.json'.format(mode_dir), 'w') as f:
            json.dump(images, f)