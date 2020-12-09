
import json
import os
import sys

from Config import get_data_dir
from Evaluate import evaluate
from Train import train

if __name__ == '__main__':
    
    # Get the chosen settings
    with open('./HPS/{}.json'.format(sys.argv[1]), 'r') as f:
        configs = json.load(f)
    os.system('rm ./HPS/{}.json'.format(sys.argv[1]))
    
    for config in configs:
        main = config['main']
        spurious = config['spurious']
        p_correct = config['p_correct']
        mode = config['mode']
        mode_param = config['mode_param']
        learning_rate = config['learning_rate']
        batch_size = config['batch_size']
        trial = config['trial']
        
        model_dir = './HPS/{}-{}-{}/{}/{}-{}/trial{}'.format(main, spurious, p_correct, mode, mode_param, learning_rate, trial)
        data_dir = '{}/{}-{}/val'.format(get_data_dir(), main, spurious)
        print(model_dir)

        train(mode, main, spurious, p_correct, trial,
                lr_override = learning_rate, mp_override = mode_param, bs_override = batch_size,
                model_dir = model_dir)
        evaluate(model_dir, data_dir)
