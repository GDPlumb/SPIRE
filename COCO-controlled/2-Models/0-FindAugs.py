
from pathlib import Path
import os
import sys

sys.path.insert(0, '../')
from Config import get_data_dir, get_split_sizes

sys.path.insert(0, '../../Common/')
from SPIRE import find_aug

if __name__ == '__main__':

    os.system('rm -rf ./FindAugs')
    os.system('mkdir ./FindAugs')

    for p_correct in [0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.975]:
        save_dir = './FindAugs/{}'.format(p_correct)
        Path(save_dir).mkdir(parents = True, exist_ok = True)

        sys.stdout = open('{}/out.txt'.format(save_dir), 'w')

        print()
        print('Finding how many images to add to each split')
        
        num_both, num_just_main, num_just_spurious, num_neither = get_split_sizes(p_correct, normalize = True)
        
        B = num_both
        M = num_just_main
        S = num_just_spurious
        N = num_neither
        
        sizes = {}
        sizes['B'] = B
        sizes['M'] = M
        sizes['S'] = S
        sizes['N'] = N

        find_aug(sizes, save_dir)
