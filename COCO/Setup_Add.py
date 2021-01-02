
import json
import numpy as np
from pathlib import Path
from PIL import Image
import sys
from torchvision import transforms

from Config import get_data_dir, get_random_seed, get_data_fold, get_max_samples

sys.path.insert(0, '../Common/')
from COCOHelper import id_from_path
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

    mask_mode = 'pixel'
    unmask = False

    np.random.seed(get_random_seed())
    
    df = get_data_fold()
    max_samples = {}
    if df == -1:
        max_samples['val'] = None
        max_samples['train'] = get_max_samples()
    else:
        max_samples['val'] = get_max_samples()
        max_samples['train'] = None

    with open('./FindSCs.json', 'r') as f:
        pairs = json.load(f)
    
    for mode in ['val', 'train']:
        mode_dir = '{}/{}'.format(get_data_dir(), mode)
        
        num_samples = max_samples[mode]
        
        with open('{}/images.json'.format(mode_dir), 'r') as f:
            images = json.load(f)
            
        coco = COCOWrapper(mode = mode)
        
        imgs = coco.get_images_with_cats(None)
        id2img = {}
        for img in imgs:
            id2img[id_from_path(img['file_name'])] = img
            
        for pair in pairs:
            main = pair.split('-')[0]
            spurious = pair.split('-')[1]
            pair_dir = '{}/{}-{}'.format(mode_dir, main, spurious)
            print(pair_dir)
            
            with open('{}/{}/splits/{}-{}.json'.format(get_data_dir(), mode, main, spurious), 'r') as f:
                splits = json.load(f)
            just_main = splits['just_main']
            just_spurious = splits['just_spurious']
            
            configs = [(just_main, just_spurious, spurious, '{}+{}'.format(main, spurious))]
            for config in configs:
        
                background_split = config[0]
                object_split = config[1]
                chosen_class = config[2]
                name = config[3]
            
                chosen_id = [coco.get_class_id(chosen_class)]
                
                save_dir = '{}/{}'.format(pair_dir, name)
                Path(save_dir).mkdir(parents = True, exist_ok = True)
            
                ids_background = [id for id in background_split]
                if num_samples is not None and num_samples < len(ids_background):
                    ids_background = np.random.choice(ids_background, size = num_samples, replace = False)
                
                ids_object = [id for id in object_split]
            
                # Get a random object to past into this background
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
                    
                    label_new = images[id]['orig'][1].copy() #Get the original labels
                    label_new[chosen_id[0]] = 1.0 #Add the object we pasted on.  Note:  this may cover other objects and so the labels are noisy
                
                    file_new = '{}/{}.jpg'.format(save_dir, id)
                
                    image_new.save(file_new)
                    images[id][name] = [file_new, label_new]
                
            with open('{}/images.json'.format(mode_dir), 'w') as f:
                json.dump(images, f)
