

import json
import os
import sys

from Config import get_data_dir
from Evaluate import evaluate
from Search import search
from Train import train

if __name__ == '__main__':
    
    # Get the chosen settings
    with open('./Models/{}.json'.format(sys.argv[1]), 'r') as f:
        configs = json.load(f)
    os.system('rm ./Models/{}.json'.format(sys.argv[1]))
    
    for config in configs:
        label1 = config['label1']
        label2 = config['label2']
        spurious = config['spurious']
        mode = config['mode']
        trial = config['trial']
        
        model_dir = './Models/{}-{}/{}/{}/trial{}'.format(label1, label2, spurious, mode, trial)
        data_dir = '{}/{}-{}/{}/val'.format(get_data_dir(), label1, label2, spurious)
        print(model_dir)
        
        #train(mode, label1, label2, spurious, trial, model_dir = model_dir)
        evaluate(model_dir, data_dir, challenge_info = (label1, label2, spurious))
        #search(model_dir, data_dir)
