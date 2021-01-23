
import json
import numpy as np
import os
import sys
import time

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
        main = config['main']
        spurious = config['spurious']
        p_correct = config['p_correct']
        mode = config['mode']
        trial = config['trial']
        
        model_dir = './Models/{}-{}/{}/{}/trial{}'.format(main, spurious, p_correct, mode, trial)
        data_dir = '{}/{}-{}/val'.format(get_data_dir(), main, spurious)
        print(model_dir)
        
        train(mode, main, spurious, p_correct, trial, model_dir = model_dir)
        evaluate(model_dir, data_dir)
        search(model_dir, data_dir)

        time.sleep(np.random.uniform(4, 6))
