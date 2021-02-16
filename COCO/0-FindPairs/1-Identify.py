
import json
import numpy as np
import os
import pickle
import sys

sys.path.insert(0, '../')
from Config import get_data_dir
from LoadImages import load_images

sys.path.insert(0, '../../Common')
from COCOHelper import id_from_path
from COCOWrapper import COCOWrapper
from Dataset import ImageDataset, my_dataloader
from FormatData import mask_images_parallel
from LoadData import load_data
from ModelWrapper import ModelWrapper
from ResNet import get_model
    
def cf_score(ids, orig_dict, cf_dict, index, min_samples = 25):
    changed = 0
    n = len(ids)
    
    if n < min_samples:
        return -1
    
    for id in ids:
        p_orig = orig_dict[id][0][index] >= 0.5
        p_cf = cf_dict[id][index] >= 0.5
        
        if p_orig != p_cf:
            changed += 1
            
    return changed / n

if __name__ == "__main__":
    
    model_file = './ReferenceModel/model.pt'
    
    mode = 'train'
    
    os.system('rm -rf Identification')
    os.system('mkdir Identification')
    
    # Setup COCOWrapper
    coco = COCOWrapper(mode = mode)
    base_dir = coco.get_base_dir()

    names = []
    for cat in coco.cats:
        names.append(cat['name'].replace(' ', '+'))
    n = len(names)

    # Setup the COCO Dataset
    data_dir = '{}/{}'.format(get_data_dir(), mode)
    images = load_images([], data_dir)
        
    ids = [key for key in images]
    filenames, labels = load_data(ids, images, ['orig'])
    dataset = ImageDataset(filenames, labels, get_names = True)
    dataloader = my_dataloader(dataset)
        
    # Load the model
    model, optim_params = get_model(mode = 'tune', parent = model_file, out_features = 91)
    model.cuda()
    model.eval()
    wrapper = ModelWrapper(model, get_names = True)

    print('Predicting Original Dataset')
    a, b, c = wrapper.predict_dataset(dataloader)
    orig_dict = {}
    for i in range(len(a)):
        id = id_from_path(c[i])
        orig_dict[id] = [a[i], b[i]]
    
    for spurious in names:
        print()
        print('Working on: ', spurious)
        index_spurious = coco.get_class_id(spurious)
    
        print('Masking Dataset')

        save_dir = '{}/tmp'.format(get_data_dir())
        os.system('rm -rf {}'.format(save_dir))
        os.system('mkdir {}'.format(save_dir))
        
        imgs_spurious = coco.get_images_with_cats([spurious.replace('+', ' ')])
        
        filenames_tmp, labels_tmp = mask_images_parallel(imgs_spurious, coco.coco, coco.get_base_dir(), save_dir, chosen_id = index_spurious, mode = 'box', unmask = True, use_png = False)

        dataset_tmp = ImageDataset(filenames_tmp, labels_tmp, get_names = True)
        dataloader_tmp = my_dataloader(dataset_tmp)
                
        print('Predicting')
        a, b, c = wrapper.predict_dataset(dataloader_tmp)
        cf_dict = {}
        for i in range(len(a)):
            id = id_from_path(c[i])
            cf_dict[id] = a[i]
            
        print('Analyzing')
        for main in names:
            if main != spurious:
                index_main = coco.get_class_id(main)
                results = {}
                
                # Get the image splits for this object pair
                with open('{}/{}/splits/{}-{}.json'.format(get_data_dir(), mode, main, spurious), 'r') as f:
                    splits = json.load(f)
                
                # Counterfactual Score evaluation
                v = cf_score(splits['both'], orig_dict, cf_dict, index_main)
                results['cf_score'] = v
                
                # Save
                with open('./Identification/{}-{}.json'.format(main, spurious), 'w') as f:
                    json.dump(results, f)
