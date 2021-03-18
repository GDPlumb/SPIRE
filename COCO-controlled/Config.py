
import sys

def get_data_dir():
    return '/home/gregory/Datasets/COCO-controlled'
    
def get_random_seed():
    return 1000

def get_split_sizes(p_correct, p_main = 0.5, p_spurious = 0.5, n = 2000, normalize = False):
    num_main = int(n * p_main)
    num_spurious = int(n * p_spurious)

    num_both = int(p_correct * num_spurious)
    num_just_main = num_main - num_both
    num_just_spurious = num_spurious - num_both
    num_neither = n - num_both - num_just_main - num_just_spurious
    
    if num_both < 0 or num_just_main < 0 or num_just_spurious < 0 or num_neither < 0:
        print('Error: Bad Distribution Setup')
        print(num_both, num_just_main, num_just_spurious, num_neither)
        sys.exit(0)
        
    if normalize:
        num_both /= n
        num_just_main /= n
        num_just_spurious /= n
        num_neither /= n
    
    return num_both, num_just_main, num_just_spurious, num_neither
