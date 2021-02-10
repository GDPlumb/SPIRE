
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
    
def get_totals(augs):
    totals = {}
    for c in CHARS:
        totals[c] = torch.tensor(0.0)
        
    for key in augs:
        source, dest = unpack_key(key)
        totals[dest] += augs[key]
        
    return totals
    
def eval_augs(sizes, deltas, augs,
            lambda_size = 100.0, lambda_diff = 10.0, lambda_remove = 1.0, lambda_add = 1.0,
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
        #if c in ['B', 'M']: # Auto-v2 differs from Auto-v1 by this single line
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

    p_main = 0.5
    p_spurious = 0.5
    n = 2000
    
    num_main = int(n * p_main)
    num_spurious = int(n * p_spurious)
    
    for p_correct in [0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.975]:

        
        save_dir = './FindAugs/{}'.format(p_correct)
        Path(save_dir).mkdir(parents = True, exist_ok = True)

        sys.stdout = open('{}/out.txt'.format(save_dir), 'w')

        print()
        print('Finding how many images to add to each split')
        
        num_both = int(p_correct * num_spurious)
        num_just_main = num_main - num_both
        num_just_spurious = num_spurious - num_both
        num_neither = n - num_both - num_just_main - num_just_spurious
        
        B = num_both / n
        M = num_just_main/ n
        S = num_just_spurious/ n
        N = num_neither / n
        
        sizes = {}
        sizes['B'] = B
        sizes['M'] = M
        sizes['S'] = S
        sizes['N'] = N
        print('Sizes')
        for c in CHARS:
            print(c, np.round(sizes[c], 3))
        print('P(Main)', np.round(B + M, 3))

        deltas = {}
        for c in CHARS:
            deltas[c] = 0.0

        # Sets P(Spurious|Main) = 0.5
        if M > B:
            diff = M - B
            deltas['B'] = diff
        else:
            diff = B - M
            deltas['M'] = diff
        
        # Restore P(Main) while setting P(Spurious|not Main) = 0.5
        P_m = B + M
        P_not_m = S + N
        target = P_not_m * (P_m + diff) / P_m
        if N > target / 2:
            deltas['S'] = target - S - N
        elif S > target / 2:
            deltas['N'] = target - S - N
        else:
            deltas['S'] = target/ 2 - S
            deltas['N'] = target / 2 - N
        
        print('Deltas')
        for c in CHARS:
            print(c, np.round(deltas[c], 3))

        # Step 2:  Find a sampling procedure to fulfill that order
        print()
        print('Finding how to sample to do that')

        augs = {}
        params = []
        for key in ['B2M', 'B2S', 'M2N', 'S2N', 'N2M', 'N2S', 'M2B', 'S2B']:
            augs[key] = torch.tensor(0.0, requires_grad = True)
            params.append(augs[key])

        opt = Adam(params, lr = 0.001)
        hist = []
        for i in range(2000):
            cost = eval_augs(sizes, deltas, augs)
            hist.append(cost.item())

            opt.zero_grad()
            cost.backward()
            opt.step()

            with torch.no_grad():
                for param in params:
                    param.clamp_(min = 0)

        plt.scatter(list(range(len(hist))), hist)
        plt.savefig('{}/eval_augs.png'.format(save_dir))
        plt.close()

        eval_augs(sizes, deltas, augs, verbose = True)

        # Summarize the results
        totals = get_totals(augs)
        sizes_aug = apply_aug(sizes, totals)

        print()
        print('Final Results')
        print('Augmented Sizes')
        for c in CHARS:
            print(c, format(sizes_aug[c]))
        p_m = (sizes_aug['B'] + sizes_aug['M']) / (sizes_aug['B'] + sizes_aug['M'] + sizes_aug['S'] + sizes_aug['N'])
        print('P(Main)', np.round(p_m.data.numpy(), 3))
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
