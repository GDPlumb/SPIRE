
from collections import defaultdict
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
    
    MAX_RATIO = 19
    
    os.system('rm -rf ./FindAugs')
    os.system('mkdir ./FindAugs')

    with open('../0-FindPairs/Pairs.json', 'r') as f:
        pairs = json.load(f)
    
    
    ###
    # Find the sampling probabilities for each pair
    ###
    
    sorted_pairs = defaultdict(list)
    summary = defaultdict(list)
    for pair in pairs:
        # Setup
        split = pair.split('-')
        main = split[0]
        spurious = split[1]
        
        save_dir = './FindAugs/{}/{}'.format(main, spurious)
        Path(save_dir).mkdir(parents = True)
        
        with open('{}/train/splits/{}.json'.format(get_data_dir(), pair), 'r') as f:
            splits = json.load(f)

        # Find the augmentation strategy
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
        
        P_S = B + S
        P_SgM = B / (B + M)
        
        bias = (P_SgM - P_S) / P_S
            
        probs = {}
        if bias >= 0:
            
            a = 1
            b = M + S
            c = M * S - B * N
            
            delta = (-1.0 * b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
            
            if delta < 0:
                delta = (-1.0 * b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
            
            if delta > B:
                delta = B
            
            ss = min(M, S)
            if delta / ss > MAX_RATIO:
                delta = MAX_RATIO * ss
            
            probs['B2M'] = delta / B
            probs['B2S'] = delta / B
            
        else:
            delta = (B * N - M * S) / (M - N)
            
            if delta > M:
                delta = M
            
            ss = min(B, S)
            if delta / ss > MAX_RATIO:
                delta = MAX_RATIO * ss
                
            probs['M2B'] = delta / M
            probs['N2S'] = delta / N
        
        # Save results
        sorted_pairs[main].append(spurious)
        
        summary['pair'].append(pair)
        for key in sizes:
            summary[key].append(sizes[key])
           
        summary['original diff'].append(B / (B + M) - S / (S + N))
        summary['delta'].append(delta)

        if bias >= 0:
            summary['aug_prob'].append(delta / B)
            summary['aug_ratio'].append(delta / ss)
            summary['new diff'].append(B / (B + M + delta) - (S + delta) / (N + S + delta))
        else:
            summary['aug_prob'].append(delta / M)
            summary['aug_ratio'].append(delta / ss)
            summary['new diff'].append((B + delta) / (B + M + delta) - (S + delta) / (S + N + delta))

        with open('{}/probs.json'.format(save_dir), 'w') as f:
            json.dump(probs, f)
            
    with open('./FindAugs/classes.json', 'w') as f:
        json.dump([key for key in sorted_pairs], f)

    with open('./FindAugs/summary.json', 'w') as f:
        json.dump(summary, f)
        
    ###
    # For each Main object, combine across each associcated Spurious object
    ###
    
    for main in sorted_pairs:
    
        save_dir = './FindAugs/{}'.format(main)
        
        names = {}
        names['orig'] = 1.0
        
        n = len(sorted_pairs[main])
        for spurious in sorted_pairs[main]:
        
            prefix = '{}-{}'.format(main, spurious)
        
            with open('{}/{}/probs.json'.format(save_dir, spurious), 'r') as f:
                probs = json.load(f)
            probs = defaultdict(int, probs)
                
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
    
    ###
    # Aggregate across all Main objects
    ###
    
    with open('./Categories.json', 'r') as f:
        cats = json.load(f)
    
    combined = {}
    for main in sorted_pairs:
        
        for cat in cats:
            if cat['name'] == main.replace('+', ' '):
                index = int(cat['id'])
                break
                
        with open('./FindAugs/{}/names.json'.format(main), 'r') as f:
            names = json.load(f)
        
        for name in names:
            if name != 'orig':
                combined[name] = [names[name], index]
        
    with open('./FindAugs/combined.json', 'w') as f:
        json.dump(combined, f)
        
