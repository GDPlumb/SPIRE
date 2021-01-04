
import json
import numpy as np
import os
import pickle
import sys

from Config import get_data_dir

sys.path.insert(0, '../Common')
from COCOHelper import id_from_path
from COCOWrapper import COCOWrapper
from Dataset import ImageDataset, my_dataloader
from FormatData import mask_images_parallel
from LoadData import load_data
from ModelWrapper import ModelWrapper
from ResNet import get_model

def splits2dist(splits):
    both = len(splits['both'])
    just_main = len(splits['just_main'])
    just_spurious = len(splits['just_spurious'])
    neither = len(splits['neither'])
    
    s_given_m = np.round(both / (both + just_main), 3)
    s_given_not_m = np.round(just_spurious / (neither + just_spurious), 3)
    
    out = {}
    out['s_given_m'] = s_given_m
    out['s_given_not_m'] = s_given_not_m
    
    return out
    
def splits2acc(splits, orig_dict, index, min_samples = 25):
    out = {}
    for name in splits:
        split = splits[name]
        n = len(split)

        pred = np.zeros((n))
        true = np.zeros((n))
        c = 0
        for id in split:
            data_tmp = orig_dict[id]
            pred[c] = data_tmp[0][index]
            true[c] = data_tmp[1][index]
            c += 1
        
        if len(pred) >= min_samples:
            v = np.mean(1 * (pred >= 0.5) == true)
        else:
            v = -1
        
        out[name] = v
        
    return out
    
def cf_score(both, orig_dict, cf_dict, index, min_samples = 20):
    changed = 0
    n = len(both)
    if n < min_samples:
        return -1
    for id in both:
        p_orig = orig_dict[id][0][index] >= 0.5
        p_cf = cf_dict[id][index] >= 0.5
                
        if p_orig != p_cf:
            changed += 1
    return changed / n

if __name__ == "__main__":

    parent = './Models/initial-transfer/trial0/model.pt'
    mode = 'train'

    os.system('rm -rf FindSCs')
    os.system('mkdir FindSCs')
    
    # Setup COCOWrapper
    coco = COCOWrapper(mode = mode)
    base_dir = coco.get_base_dir()

    names = []
    for cat in coco.cats:
        names.append(cat['name'].replace(' ', '+'))
    n = len(names)

    # Setup the COCO Dataset
    data_dir = '{}/{}'.format(get_data_dir(), mode)
    with open('{}/images.json'.format(data_dir), 'r') as f:
        images = json.load(f)
    ids = [key for key in images]
    filenames, labels = load_data(ids, images, ['orig'])
    dataset = ImageDataset(filenames, labels, get_names = True)
    dataloader = my_dataloader(dataset)
        
    # Load the model
    model, optim_params = get_model(mode = 'tune', parent = parent, out_features = 91)
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
        
        mask_images_parallel(imgs_spurious, coco.coco, coco.get_base_dir(), save_dir, chosen_id = index_spurious, mode = 'box', unmask = True, use_png = False)

        with open('{}-info.p'.format(save_dir), 'rb') as f:
            filenames_tmp, labels_tmp = pickle.load(f)
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
                    
                # Distribution evaluation
                out = splits2dist(splits)
                for key in out:
                    results[key] = out[key]
                
                # Accuracy per Split evalaution
                out = splits2acc(splits, orig_dict, index_main)
                for key in out:
                    results[key] = out[key]
                
                # Counterfactual Score evaluation
                results['cf_score'] = cf_score(splits['both'], orig_dict, cf_dict, index_main)
                
                # Save
                with open('./FindSCs/{}-{}.json'.format(main, spurious), 'w') as f:
                    json.dump(results, f)
