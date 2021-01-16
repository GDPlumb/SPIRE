
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import sys
import torch
from torch.optim import Adam

from Config import get_data_dir

CHARS = ['B', 'M', 'S', 'N']

def format(v):
    return np.round(v.data.numpy(), 3)

def unpack_key(key):
    split = key.split('2')
    source = split[0]
    dest = split[1]
    return source, dest

def apply_aug(sizes, deltas):
    out = {}
    for c in CHARS:
        out[c] = sizes[c] + deltas[c]
    return out
    
def dist_metrics(sizes):
    
    B = sizes['B']
    M = sizes['M']
    S = sizes['S']
    N = sizes['N']
    
    P_M = (B + M) / (B + M + S + N)
    P_S = (B + S) / (B + M + S + N)
    
    P_S_given_M = B / (B + M)
    P_S_given_not_M = S / (S + N)

    return P_M, P_S, P_S_given_M, P_S_given_not_M

def eval_1(sizes, deltas,
        T_uses_S = False,
        lambda_main = 2, lambda_ind = 2, lambda_reg = 0.001,
        verbose = False):
    
    if verbose:
        print('Deltas')
        for c in CHARS:
            print(c, format(deltas[c]))

    P_M, P_S, _, _ = dist_metrics(sizes)
    if T_uses_S:
        T = P_S
    else:
        T = torch.tensor(0.5)

    sizes_aug = apply_aug(sizes, deltas)
    if verbose:
        print('Augmented Sizes')
        for c in CHARS:
            print(c, format(sizes_aug[c]))
    
    P_M_new, _, P_S_given_M, P_S_given_not_M = dist_metrics(sizes_aug)
    
    if verbose:
        print('Analysis')
        
    cost = torch.tensor(0.0)
    
    v = P_M_new - P_M
    if verbose:
        print('P(M) bias', format(v))
    cost += lambda_main * torch.abs(v)
    
    v = P_S_given_M - T
    if verbose:
        print('P(S | M) bias', format(v))
    cost += lambda_ind * torch.abs(v)
 
    v = P_S_given_not_M - T
    if verbose:
        print('P(S | !M) bias', format(v))
    cost += lambda_ind * torch.abs(v)
    
    cost += lambda_reg * (deltas['B'] + deltas['N'])
 
    if verbose:
        print('Cost', format(cost))
    
    return cost
    
def get_totals(augs):
    totals = {}
    for c in CHARS:
        totals[c] = torch.tensor(0.0)
        
    for key in augs:
        source, dest = unpack_key(key)
        totals[dest] += augs[key]
        
    return totals
    
def eval_2(sizes, deltas, augs,
            lambda_size = 100.0, lambda_diff = 5.0, lambda_remove = 0.1, lambda_add = 0.1,
            verbose = False):
            
    if verbose:
        print('Augs')
        for key in augs:
            print(key, format(augs[key]))

    cost = torch.tensor(0.0)
    
    # Penalize any illegal sizes
    for key in augs:
        source, dest = unpack_key(key)
        dif = augs[key] - sizes[source]
        cost += lambda_size * dif.clamp_(min = 0.0)
        
    # Penalize errors in meeting the deltas
    totals = get_totals(augs)
    if verbose:
        print('Totals')
        for c in CHARS:
            print(c, format(totals[c]))
        
    for c in CHARS:
        if c in ['B', 'M']: # Auto-v2 differs from Auto-v1 by this single line
            cost += lambda_diff * torch.abs(totals[c] - deltas[c])
        
    # Check that the intervention does not leave traces
    v = augs['B2M'] - (augs['B2S'] + augs['M2N'] + augs['S2N'])
    if verbose:
        print('Removing Bias', format(v))
    cost += lambda_remove * torch.abs(v)
    
    v = augs['N2S'] - (augs['N2M'] + augs['M2B'] + augs['S2B'])
    if verbose:
        print('Adding Bias', format(v))
    cost += lambda_add * torch.abs(v)
    
    if verbose:
        print('Cost', format(cost))
        
    return cost

if __name__ == '__main__':

    os.system('rm -rf ./FindAugs')
    os.system('mkdir ./FindAugs')

    with open('./FindSCs.json', 'r') as f:
        pairs = json.load(f)
    
    sorted_pairs = {}
    for pair in pairs:
        # Setup this pair
        split = pair.split('-')
        main = split[0]
        spurious = split[1]
        
        if main in sorted_pairs:
            sorted_pairs[main].append(spurious)
        else:
            sorted_pairs[main] = [spurious]
        
        save_dir = './FindAugs/{}/{}'.format(main, spurious)
        Path(save_dir).mkdir(parents = True, exist_ok = True)

        sys.stdout = open('{}/out.txt'.format(save_dir), 'w')

        # Step 1:  Find how many images to add to each split
        with open('{}/train/splits/{}.json'.format(get_data_dir(), pair), 'r') as f:
            splits = json.load(f)

        n = 0
        for key in splits:
            splits[key] = len(splits[key])
            n += splits[key]

        sizes = {}
        sizes['B'] = torch.tensor(splits['both'] / n)
        sizes['M'] = torch.tensor(splits['just_main'] / n)
        sizes['S'] = torch.tensor(splits['just_spurious'] / n)
        sizes['N'] = torch.tensor(splits['neither'] / n)

        print()
        print('Finding how many images to add to each split')
        print('Sizes')
        for c in CHARS:
            print(c, format(sizes[c]))

        deltas = {}
        params = []
        for i, v in enumerate([0.0, 0.0, 0.0, 0.0]):
            deltas[CHARS[i]] = torch.tensor(v, requires_grad = True)
            params.append(deltas[CHARS[i]])

        opt = Adam(params, lr = 0.0005)
        hist = []
        for i in range(3000):
            cost = eval_1(sizes, deltas)
            hist.append(cost.item())

            opt.zero_grad()
            cost.backward()
            opt.step()

            with torch.no_grad():
                for param in params:
                    param.clamp_(min = 0)

        plt.scatter(list(range(len(hist))), hist)
        plt.savefig('{}/Step1.png'.format(save_dir))
        plt.close()

        eval_1(sizes, deltas, verbose = True)

        # Step 2:  Find a sampling procedure to fulfill that order
        print()
        print('Finding how to sample to do that')
        for c in CHARS:
            deltas[c].requires_grad = False

        augs = {}
        params = []
        for key in ['B2M', 'B2S', 'M2N', 'S2N', 'N2M', 'N2S', 'M2B', 'S2B']:
            augs[key] = torch.tensor(0.0, requires_grad = True)
            params.append(augs[key])

        opt = Adam(params, lr = 0.001)
        hist = []
        for i in range(2000):
            cost = eval_2(sizes, deltas, augs)
            hist.append(cost.item())

            opt.zero_grad()
            cost.backward()
            opt.step()

            with torch.no_grad():
                for param in params:
                    param.clamp_(min = 0)

        plt.scatter(list(range(len(hist))), hist)
        plt.savefig('{}/Step2.png'.format(save_dir))
        plt.close()

        eval_2(sizes, deltas, augs, verbose = True)

        # Summarize the results
        totals = get_totals(augs)
        sizes_aug = apply_aug(sizes, totals)

        print()
        print('Final Results')
        print('Augmented Sizes')
        for c in CHARS:
            print(c, format(sizes_aug[c]))
        print('Sampling Probabilities')
        probs = {}
        for key in augs:
            source, dest = unpack_key(key)
            v = augs[key] / max(sizes[source], 1e-8)
            print(key, format(v))
            probs[key] = float(v.data.numpy())
        print()

        with open('{}/probs.json'.format(save_dir), 'w') as f:
            json.dump(probs, f)
    
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
                names['{}-both-spurious-box'.format(prefix)] = v / n

            v = probs['B2S']
            if v != 0.0:
                names['{}-both-main-box'.format(prefix)] = v / n

            v = probs['M2N']
            if v != 0.0:
                names['{}-just_main-main-box'.format(prefix)] = v / n

            v = probs['S2N']
            if v != 0.0:
                names['{}-just_spurious-spurious-box'.format(prefix)] = v / n

            v = probs['N2M']
            if v != 0.0:
                names['{}-neither+main'.format(prefix)] = v / n

            v = probs['N2S']
            if v != 0.0:
                names['{}-neither+spurious'.format(prefix)] = v / n

            v = probs['M2B']
            if v != 0.0:
                names['{}-just_main+spurious'.format(prefix)] = v / n

            v = probs['S2B']
            if v != 0.0:
                names['{}-just_spurious+main'.format(prefix)] = v / n
        
        with open('{}/names.json'.format(save_dir), 'w') as f:
            json.dump(names, f)
