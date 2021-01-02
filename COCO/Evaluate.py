
import json
import numpy as np
import sys

sys.path.insert(0, '../Common/')
from COCOHelper import id_from_path
from Dataset import ImageDataset, my_dataloader
from LoadData import load_data
from ModelWrapper import ModelWrapper
from ResNet import get_model

def evaluate(model_dir, data_dir, coco, min_samples = 25):

    # Load the needed information
    with open('{}/images.json'.format(data_dir), 'r') as f:
        images = json.load(f)
    
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
                v = -1
            
            out['{}-{}'.format(pair, split_name)] = v
    
    with open('{}/results.json'.format(model_dir), 'w') as f:
        json.dump(out, f)
