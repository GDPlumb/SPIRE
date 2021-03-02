
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, '../')
from Config import get_data_dir

sys.path.insert(0, '../../Common/')
from COCOWrapper import COCOWrapper, id_from_path
from FormatData import mask_images_parallel

if __name__ == '__main__':
    
    # Configuration
    label1 = 'runway'
    label2 = 'street'
    spurious = 'airplane'
    
    tuple_dir = '{}/{}-{}/{}'.format(get_data_dir(), label1, label2, spurious)
    
    for mode in ['val', 'train']:
        mode_dir = '{}/{}'.format(tuple_dir, mode)

        # Setup COCO
        coco = COCOWrapper(mode = mode)
        coco.construct_id2img()
            
        # Load the Splits
        with open('{}/splits.json'.format(mode_dir), 'r') as f:
            splits = json.load(f)
        
        # Get which object is being removed
        chosen_class = spurious
        chosen_id = coco.get_class_id(chosen_class)
        
        class_type = 'spurious'
        unmask = False
        unmask_classes = None
        
        # Create the counterfactual images
        configs = [('1s', 1), ('0s', 0)]
        for config in configs:
            # Get the base images
            name = config[0]
            imgs = coco.get_imgs_by_ids(splits[name])

            # Get the label
            label = config[1]
            
            config_dir = '{}/{}-{}'.format(mode_dir, name, class_type)
            for mask_mode in ['box', 'pixel']:
                # Setup the output directory
                save_dir = '{}/{}'.format(config_dir, mask_mode)
                os.system('rm -rf {}'.format(save_dir))
                Path(save_dir).mkdir(parents = True)
                print(save_dir)

                # Maks the chosen object
                filenames, _ = mask_images_parallel(imgs, coco, 
                                                    save_dir,
                                                    chosen_id = chosen_id, mode = mask_mode,
                                                    unmask = unmask, unmask_classes = None,
                                                    use_png = (mask_mode == 'pixel'))
                
                # Save the output
                images = {}
                for i in range(len(filenames)):
                    filename = filenames[i]
                    id = id_from_path(filename)
                    images[id] = [filename, label]

                with open('{}/images.json'.format(save_dir), 'w') as f:
                    json.dump(images, f)
                