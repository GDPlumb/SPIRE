

import json
import os
import sys

from Config import get_data_dir, get_data_fold
from Evaluate import evaluate
from Search import search
from Train import train

sys.path.insert(0, '../Common/')
from COCOWrapper import COCOWrapper

if __name__ == '__main__':
    
    # Get the chosen settings
    with open('./Models/{}.json'.format(sys.argv[1]), 'r') as f:
        configs = json.load(f)
    os.system('rm ./Models/{}.json'.format(sys.argv[1]))
    
    df = get_data_fold()
    if df == -1:
        fold = 'train'
    else:
        fold = 'val'
    
    for config in configs:
        mode = config['mode']
        trial = config['trial']
        
        data_dir = '{}/{}'.format(get_data_dir(), fold)
        model_dir = './Models/{}/trial{}'.format(mode, trial)
        
        print(model_dir, data_dir)

        coco = COCOWrapper(mode = fold)

        #train(mode, trial, model_dir = model_dir)
        evaluate(model_dir, data_dir, coco)
        search(model_dir, data_dir, coco)
