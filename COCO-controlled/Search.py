
import json
import pickle
import sys
import torch
import torchvision.models as models

from Config import get_data_dir
from Misc import id_from_path, load_data

sys.path.insert(0, '../COCO/')
from Dataset import ImageDataset, my_dataloader
from ModelWrapper import ModelWrapper

def search(mode, main, spurious, p_correct, trial):

    base = './Models/{}-{}/{}/{}/trial{}'.format(main, spurious, p_correct, mode, trial)
    
    # Load the images for this pair
    data_dir = '{}/{}-{}/val'.format(get_data_dir(), main, spurious)
    with open('{}/splits.p'.format(data_dir), 'rb') as f:
        splits = pickle.load(f)
    ids = splits['both']
    
    with open('{}/images.p'.format(data_dir), 'rb') as f:
        images = pickle.load(f)
        
    # Setup the model
    model = models.mobilenet_v2(pretrained = True)
    model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 1)
    model.cuda()
    
    model.load_state_dict(torch.load('{}/model.pt'.format(base)))
    model.eval()
    
    wrapper = ModelWrapper(model, get_names = True)
    
    # Get the model's predictions on relevant images
    pred_map = {}
    for name in ['orig', 'box-spurious', 'pixel-spurious-paint', 'box-main', 'pixel-main-paint']:
        tmp = {}
    
        files_tmp, labels_tmp = load_data(ids, images, [name])
        
        dataset_tmp = ImageDataset(files_tmp, labels_tmp, get_names = True)
        dataloader_tmp = my_dataloader(dataset_tmp)
        
        y_hat, y_true, files = wrapper.predict_dataset(dataloader_tmp)
        
        for i in range(len(y_hat)):
            tmp[id_from_path(files[i])] = (1 * (y_hat[i] >= 0.5))[0]
        
        pred_map[name] = tmp
        
    # Aggregate the results
    out = {}
    
    for name in ['box-spurious', 'pixel-spurious-paint', 'box-main', 'pixel-main-paint']:
        matrix = [[0,0],[0,0]]
        for id in ids:
            p_orig = pred_map['orig'][id]
            p_new = pred_map[name][id]
            matrix[p_orig][p_new] += 1
        out[name] = matrix
        
    with open('{}/search.json'.format(base), 'w') as f:
        json.dump(out, f)

if __name__ == '__main__':

    mode = sys.argv[1]
    main = sys.argv[2]
    spurious = sys.argv[3]
    p_correct = float(sys.argv[4])
    trials = sys.argv[5].split(',')

    for trial in trials:
        search(mode, main, spurious, p_correct, trial)
