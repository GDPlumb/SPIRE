
import glob
import json
import numpy as np
import sys

from LoadImages import load_images

sys.path.insert(0, '../Common/')
from COCOHelper import id_from_path
from Dataset import ImageDataset, my_dataloader
from LoadData import load_data
from ModelWrapper import ModelWrapper
from ResNet import get_model

def evaluate(model_dir, data_dir, coco, min_samples = 25):

    # Load the needed information
    images = load_images([], data_dir)
    
    with open('./FindSCs.json', 'r') as f:
        pairs = json.load(f)
    
    # Setup the model
    model = get_model(mode = 'eval', parent = '{}/model.pt'.format(model_dir), out_features = 91)
    model.cuda()
    model.eval()
    
    wrapper = ModelWrapper(model, get_names = True)
    
    # Get the predictions for all of dataset
    ids = [id for id in images]
    files, labels = load_data(ids, images, ['orig'])
    dataset = ImageDataset(files, labels, get_names = True)
    dataloader = my_dataloader(dataset)
    y_hat, y_true, names = wrapper.predict_dataset(dataloader)
    
    data_map = {}
    for i, name in enumerate(names):
        data_map[id_from_path(name)] = [y_hat[i], y_true[i]]
                
    # Run the evaluation
    out = {}
    
    precision, recall = wrapper.metrics(y_hat, y_true)
    map, mar = coco.get_metrics(precision, recall)
    out['MAP'] = map
    out['MAR'] = mar
    
    for pair in pairs:
        main = pair.split('-')[0]
        spurious = pair.split('-')[1]
        
        # Get the index that we care about for this pair
        index = coco.get_class_id(main)
        
        # Get the image splits for this pair
        with open('{}/splits/{}-{}.json'.format(data_dir, main, spurious), 'r') as f:
            splits = json.load(f)
        
        # Calculate the accuracy for each split
        splits_acc = {}
        for split_name in splits:
            split = splits[split_name]
            n = len(split)

            pred = np.zeros((n))
            true = np.zeros((n))
            c = 0
            for id in split:
                data_tmp = data_map[id]
                pred[c] = data_tmp[0][index]
                true[c] = data_tmp[1][index]
                c += 1
            
            if len(pred) >= min_samples:
                v = np.mean(1 * (pred >= 0.5) == true)
            else:
                # Load the ChallengeSet
                files_cs = glob.glob('./ChallengeSets/{}-{}/*'.format(pair, split_name))
                n_cs = len(files_cs)
                if n_cs == 0:
                    print(pair, split_name)
                    
                if split_name in ['both', 'just_main']:
                    labels_cs = np.ones((n_cs), dtype = np.float32)
                elif split_name in ['just_spurious', 'neither']:
                    labels_cs = np.zeros((n_cs), dtype = np.float32)
                
                dataset_cs = ImageDataset(files_cs, labels_cs, get_names = True)
                dataloader_cs = my_dataloader(dataset_cs)
                y_hat_cs, y_true_cs, names_cs = wrapper.predict_dataset(dataloader_cs)
                
                correct = 0
                for i in range(n_cs):
                    if 1 * (y_hat_cs[i][index] >= 0.5) == y_true_cs[i]:
                        correct += 1
                v = correct / n_cs
            
            out['{}-{}'.format(pair, split_name)] = v
            splits_acc[split_name] = v
        
        # Get some common information for computing metrics
        for key in splits:
            splits[key] = len(splits[key])
        total = splits['both'] + splits['just_main'] + splits['just_spurious'] + splits['neither']
        p_main = (splits['both'] + splits['just_main']) / total
        
        both = splits_acc['both']
        just_main = splits_acc['just_main']
        just_spurious = splits_acc['just_spurious']
        neither = splits_acc['neither']

        # Compute the 'balanced' precision and recall
        # -  We keep P(Main) from the original dataset
        # -  We then construct a distribution where Spurious is independent from Main
        # -  We use either 1/2 or P(Spurious) from the original dataset while doing this
        p_spurious = 0.5 #(splits['both'] + splits['just_spurious']) / total
        tp = p_main * (p_spurious * both + (1 - p_spurious) * just_main)
        fp = (1 - p_main) * (p_spurious * (1 - just_spurious)  + (1 - p_spurious) * (1 -  neither))
        precision = tp / (tp + fp + 1e-16)
        out['{}-b-precision'.format(pair)] = precision
        out['{}-b-recall'.format(pair)] = p_spurious * both + (1 - p_spurious) * just_main
        
        # Compute r-gap (p-gap is hard to interpret when calculated correctly)
        out['{}-r-gap'.format(pair)] = abs(both - just_main)

    with open('{}/results.json'.format(model_dir), 'w') as f:
        json.dump(out, f)
