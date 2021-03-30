
import json
import numpy as np
import os
from pathlib import Path
import sys

sys.path.insert(0, '../')
from Config import get_data_dir

sys.path.insert(0, '../../Common/')
from SPIRE import find_aug

def format(v):
    return np.round(v.data.numpy(), 3)

if __name__ == '__main__':

    os.system('rm -rf ./FindAugs')
    os.system('mkdir ./FindAugs')

    with open('../0-FindPairs/Pairs.json', 'r') as f:
        pairs = json.load(f)
    
    sorted_pairs = {}
    for pair in pairs:
        split = pair.split('-')
        main = split[0]
        spurious = split[1]
        
        if main in sorted_pairs:
            sorted_pairs[main].append(spurious)
        else:
            sorted_pairs[main] = [spurious]
        
        save_dir = './FindAugs/{}/{}'.format(main, spurious)
        Path(save_dir).mkdir(parents = True)

        sys.stdout = open('{}/out.txt'.format(save_dir), 'w')

        print()
        print('Finding how many images to add to each split')
        
        with open('{}/train/splits/{}.json'.format(get_data_dir(), pair), 'r') as f:
            splits = json.load(f)

        n = 0
        for key in splits:
            splits[key] = len(splits[key])
            n += splits[key]
        
        B = splits['both'] / n
        M = splits['just_main'] / n
        S = splits['just_spurious'] / n
        N = splits['neither'] / n
        
        sizes = {}
        sizes['B'] = B
        sizes['M'] = M
        sizes['S'] = S
        sizes['N'] = N
        
        find_aug(sizes, save_dir)


    with open('./FindAugs/classes.json', 'w') as f:
        json.dump([key for key in sorted_pairs], f)

    for main in sorted_pairs:
    
        save_dir = './FindAugs/{}'.format(main)
        
        names = {}
        names['orig'] = 1.0
        
        n = len(sorted_pairs[main])
        for spurious in sorted_pairs[main]:
        
            prefix = '{}-{}'.format(main, spurious)
        
            with open('{}/{}/probs.json'.format(save_dir, spurious), 'r') as f:
                probs = json.load(f)
                
            v = probs['B2M']
            if v != 0.0:
                names['{}/both-spurious/box'.format(prefix)] = v / n

            v = probs['B2S']
            if v != 0.0:
                names['{}/both-main/box'.format(prefix)] = v / n

            v = probs['M2N']
            if v != 0.0:
                names['{}/just_main-main/box'.format(prefix)] = v / n

            v = probs['S2N']
            if v != 0.0:
                names['{}/just_spurious-spurious/box'.format(prefix)] = v / n

            v = probs['N2M']
            if v != 0.0:
                names['{}/neither+main'.format(prefix)] = v / n

            v = probs['N2S']
            if v != 0.0:
                names['{}/neither+spurious'.format(prefix)] = v / n

            v = probs['M2B']
            if v != 0.0:
                names['{}/just_main+spurious'.format(prefix)] = v / n

            v = probs['S2B']
            if v != 0.0:
                names['{}/just_spurious+main'.format(prefix)] = v / n
        
        with open('{}/names.json'.format(save_dir), 'w') as f:
            json.dump(names, f)
