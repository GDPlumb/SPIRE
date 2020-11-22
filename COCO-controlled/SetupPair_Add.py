
import numpy as np
import os
import pickle
from PIL import Image
import sys
from torchvision import transforms

from Config import get_data_dir, get_random_seed
from Misc import id_from_path

sys.path.insert(0, '../COCO/')
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

    main = sys.argv[1]
    spurious = sys.argv[2]
    
    mask_mode = 'pixel'
    unmask = False
    
    np.random.seed(get_random_seed())
    
    pair_dir = '{}/{}-{}'.format(get_data_dir(), main, spurious)
    for mode in ['val', 'train']:
        mode_dir = '{}/{}'.format(pair_dir, mode)
        
        with open('{}/images.p'.format(mode_dir), 'rb') as f:
            images = pickle.load(f)
            
        with open('{}/splits.p'.format(mode_dir), 'rb') as f:
            splits = pickle.load(f)
            
        coco = COCOWrapper(mode = mode)
        
        imgs = coco.get_images_with_cats(None)
        id2img = {}
        for img in imgs:
            id2img[id_from_path(img['file_name'])] = img

        for config in [('just_spurious', 'just_main', main, 1), \
                        ('just_main', 'just_spurious', spurious, 1), \
                        ('neither', 'just_main', main, 1), \
                        ('neither', 'just_spurious', spurious, 0)]:
        
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
                if mode == 'train':
                    # On the training distributions, Just Main and Just Spurious have the same size and we draw the first k samples from the list to use for each split
                    # As a result, we need to be careful not to leak extra information by seeing objects drawn from images that are not part of those k samples
                    indices[i] = i
                elif mode == 'val':
                    # On the validation set, Just Main and Just Spurious have different sides
                    # But we do not need to worry about data leaking, so we pick a random source
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
            
