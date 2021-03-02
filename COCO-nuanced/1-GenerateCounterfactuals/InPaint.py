
import os
import json
from pathlib import Path
import skimage
import sys

sys.path.insert(0, '../')
from Config import get_data_dir

sys.path.insert(0, '../../Common/')
from FormatData_InPaint import InPaintWrapper

if __name__ == '__main__':

    # Configuration
    label1 = 'runway'
    label2 = 'street'
    spurious = 'airplane'
    
    tuple_dir = '{}/{}-{}/{}'.format(get_data_dir(), label1, label2, spurious)
    
    painter = InPaintWrapper()
    
    for mode in ['val', 'train']:
        mode_dir = '{}/{}'.format(tuple_dir, mode)

        # Create the InPainted Images
        configs = ['0s-spurious/pixel', '1s-spurious/pixel']
        for config in configs:
            save_dir = '{}/{}-paint'.format(mode_dir, config)
            os.system('rm -rf {}'.format(save_dir))
            Path(save_dir).mkdir(parents = True)
            print(save_dir)
            
            with open('{}/{}/images.json'.format(mode_dir, config), 'r') as f:
                images = json.load(f)
                
            ids = [id for id in images]
                      
            filenames = []
            labels = []
            for id in ids:
                filenames.append(images[id][0])
                labels.append(images[id][1])
                
            images_painted = painter.inpaint(filenames)
                      
            # Save the output
            images = {}
            for i, id in enumerate(ids):
                filename = filenames[i].split('/')[-1][:-3] + 'jpg'
                image = images_painted[i]
                label = labels[i]
                
                images[id] = [filename, label]
                skimage.io.imsave('{}/{}'.format(save_dir, filename), image)
            
            with open('{}/images.json'.format(save_dir), 'w') as f:
                json.dump(images, f)
                