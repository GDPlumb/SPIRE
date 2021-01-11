
import json
import numpy as np
import os
from pathlib import Path
import pickle
from PIL import Image
import sys

from Config import get_data_dir, get_random_seed, get_data_fold, get_max_samples

sys.path.insert(0, '../Common/')
from COCOHelper import id_from_path
from COCOWrapper import COCOWrapper
from FormatData import merge_images_parallel

if __name__ == '__main__':

    print('Adding')

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
            
        for pair in pairs:
            main = pair.split('-')[0]
            spurious = pair.split('-')[1]
            pair_dir = '{}/{}-{}'.format(mode_dir, main, spurious)
            print(pair_dir)
            
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
                
            type2name = {}
            type2name['main'] = main
            type2name['spurious'] = spurious
            
            configs = [('neither', 'main'), ('neither', 'spurious'), ('just_main', 'spurious'), ('just_spurious', 'main')]
            for config in configs:
        
                background_name = config[0]
                background_split = splits[background_name]
                
                ids_background = [id for id in background_split]
                if num_samples is not None and num_samples < len(ids_background):
                    ids_background = np.random.choice(ids_background, size = num_samples, replace = False)
                
                class_name = config[1]
                chosen_class = type2name[class_name]
                chosen_id = [coco.get_class_id(chosen_class)]
                
                ids_object_all = [id for id in splits[class_name]]
                n = len(ids_object_all)
                
                # Get a random object to past into each background
                ids_object = []
                for i in range(len(ids_background)):
                    ids_object.append(ids_object_all[np.random.randint(0, n)])
                    
                name = '{}-{}-{}+{}'.format(main, spurious, background_name, class_name)
                
                save_dir = '{}/{}+{}'.format(pair_dir, background_name, class_name)
                Path(save_dir).mkdir(parents = True, exist_ok = True)
                    
                # Merge the iimages
                merge_images_parallel(coco, save_dir, ids_background, ids_object, chosen_id)
                
                # Save the results
                with open('{}-info.p'.format(save_dir), 'rb') as f:
                    filenames, labels = pickle.load(f)
                os.system('rm {}-info.p'.format(save_dir))

                for i in range(len(filenames)):
                    filename = filenames[i]
                    label = list(np.copy(labels[i]))
                    id = id_from_path(filename)

                    images[id][name] = [filename, label]

                with open('{}/images.json'.format(mode_dir), 'w') as f:
                    json.dump(images, f)
                
