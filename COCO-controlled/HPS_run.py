
import json
import os
import sys

from Config import get_data_dir
from Evaluate import evaluate
from Train import train

if __name__ == '__main__':
    
    # Get the chosen settings
    with open('./HPS/{}.json'.format(sys.argv[1]), 'r') as f:
        config = json.load(f)
    os.system('rm ./HPS/{}.json'.format(sys.argv[1]))
        
    main = config['main']
    spurious = config['spurious']
    p_correct = config['p_correct']
    
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    mode = config['mode']
    mode_param = config['mode_param']
    
    trial = config['trial']
    
    base = './HPS/{}-{}-{}/{}/{}-{}/trial{}'.format(main, spurious, p_correct, mode, mode_param, learning_rate, trial)

    # Train the model
    train(mode, main, spurious, p_correct, trial,
            lr_override = learning_rate, mp_override = mode_param,
            base = base)

    # Evaluate the model
    evaluate(base, '{}/{}-{}/val'.format(get_data_dir(), main, spurious))
