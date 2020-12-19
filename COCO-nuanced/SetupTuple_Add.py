
import json
import numpy as np
import os
import pickle
from PIL import Image
import sys
from torchvision import transforms

from Config import get_data_dir, get_random_seed
from Misc import id_from_path

sys.path.insert(0, '../Common/')
from COCOWrapper import COCOWrapper
from Dataset import MakeSquare
from FormatData import get_mask

def get_custom_resize(d):
    return transforms.Compose([
            MakeSquare(),
            transforms.Resize((d,d))
            ])

if __name__ == '__main__':

    print('Adding')

    label1 = sys.argv[1]
    label2 = sys.argv[2]
    spurious = sys.argv[3]
    tuple_dir = '{}/{}-{}/{}'.format(get_data_dir(), label1, label2, spurious)
    
    mask_mode = 'pixel'
    unmask = False
    
    np.random.seed(get_random_seed())
    
    for mode in ['val', 'train']:
        mode_dir = '{}/{}'.format(tuple_dir, mode)
        
        with open('{}/images.p'.format(mode_dir), 'rb') as f:
            images = pickle.load(f)
            
        with open('{}/splits.json'.format(mode_dir), 'rb') as f:
            splits = json.load(f)
            
        coco = COCOWrapper(mode = mode)
        
        imgs = coco.get_images_with_cats(None)
        id2img = {}
        for img in imgs:
            id2img[id_from_path(img['file_name'])] = img

        for config in [('1ns', '1s', spurious, 1), \
                        ('0ns', '1s', spurious, 0)]:
        
            background_split = config[0]
            object_split = config[1]
            chosen_class = config[2]
            label = config[3]
            
            chosen_id = [coco.get_class_id(chosen_class)]
            
            save_location = '{}/{}+{}'.format(mode_dir, background_split, object_split)
            os.system('rm -rf {}'.format(save_location))
            os.system('mkdir {}'.format(save_location))
            
            ids_background = splits[background_split]
            ids_object = splits[object_split]
            
            n = len(ids_background)
            indices = np.zeros((n), dtype = np.int)
            for i in range(n):
                indices[i] = np.random.randint(0, len(ids_object))
                    
            for i in range(n):
            
                id = ids_background[i]
                
                id_object = ids_object[indices[i]]
                
                anns_object = coco.coco.loadAnns(coco.coco.getAnnIds(imgIds = id2img[id_object]['id']))
                mask = get_mask(anns_object, chosen_id, coco.coco, mode = mask_mode, unmask = unmask)
                                
                object_image = Image.open(images[id_object]['orig'][0]).convert('RGB')
                
                base_image = np.array(Image.open(images[id]['orig'][0]).convert('RGB'))
                width, height, _ = base_image.shape
                dim_min = min(width, height)
                
                custom_resize = get_custom_resize(dim_min)
                
                mask = np.array(custom_resize(Image.fromarray(np.squeeze(mask))))
                object_image = np.array(custom_resize(object_image))
                
                mask_indices = np.where(mask != 0)

                for j in range(3):
                    base_image[mask_indices[0], mask_indices[1], j] = object_image[mask_indices[0], mask_indices[1], j]

                image_new = Image.fromarray(np.uint8(base_image))
                
                file_new = '{}/{}.jpg'.format(save_location, id)
                
                image_new.save(file_new)
                images[id]['{}+{}'.format(background_split, object_split)] = [file_new, label]
                
        with open('{}/images.p'.format(mode_dir), 'wb') as f:
            pickle.dump(images, f)
            

