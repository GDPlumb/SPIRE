
import json
import os
from pathlib import Path
import sys
import torch

sys.path.insert(0, '../Common')
from ResNet import get_model

for trial in [0, 1, 2, 3]:

    model = get_model(mode = 'eval', parent = './Models/initial-tune/trial{}/model.pt'.format(trial), out_features = 91)

    with open('./COCO_cats.json', 'r') as f: #This is a json copy of coco.loadCats(coco.getCatIds())
        cats = json.load(f)

    with open('./FindSCs.json', 'r') as f:
        pairs = json.load(f)
    pairs = [key for key in pairs]
    
    for i, pair in enumerate(pairs):
        main = pair.split('-')[0].replace('+', ' ')
        
        for cat in cats:
            if cat['name'] == main.replace('+', ' '):
                index = int(cat['id'])
                break

        model_partial = get_model(mode = 'eval', parent = './Models/partial-{}-transfer/trial{}/model.pt'.format(i, trial), out_features = 91)
        
        model.fc.bias[index] = model_partial.fc.bias[index]
        model.fc.weight[index, :] = model_partial.fc.weight[index, :]

    save_dir = './Models/corrected/trial{}'.format(trial)
    os.system('rm -rf {}'.format(save_dir))
    Path(save_dir).mkdir(parents = True, exist_ok = True)

    torch.save(model.state_dict(), '{}/model.pt'.format(save_dir))
