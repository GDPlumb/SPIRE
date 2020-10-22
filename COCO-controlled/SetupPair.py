
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import pickle
from PIL import Image
import random
import sys

from Misc import get_pair

sys.path.insert(0, '../COCO/')
from COCOWrapper import COCOWrapper

if __name__ == '__main__':

    main = sys.argv[1]
    spurious = sys.argv[2]
    
    root = '/home/gregory/Datasets/COCO'
    year = '2017'
    num_sample = 1000

    # Get the image splits for the training data
    coco = COCOWrapper(root = root, mode = 'train', year = year)
    both, just_main, just_spurious, neither = get_pair(coco, main, spurious)
    
    # Randomly chose num_sample from each of the splits to use for training
    both_chosen = random.sample(list(both), num_sample)
    just_main_chosen = random.sample(list(just_main), num_sample)
    just_spurious_chosen = random.sample(list(just_spurious), num_sample)
    neither_chosen = random.sample(list(neither), num_sample)
    
    # Setup the directory and save the output
    name = './Pairs/{}-{}'.format(main, spurious)
    os.system('rm -rf {}'.format(name))
    os.system('mkdir {}'.format(name))
    
    with open('{}/splits.p'.format(name), 'wb') as f:
        pickle.dump([both_chosen, just_main_chosen, just_spurious_chosen, neither_chosen], f)
    
    # Check the splits look right
    im_array = []
    for file_name in both_chosen[:5]:
        im_array.append(Image.open('{}/train{}/{}'.format(root, year, file_name)))
        
    for file_name in just_main_chosen[:5]:
        im_array.append(Image.open('{}/train{}/{}'.format(root, year, file_name)))

    for file_name in just_spurious_chosen[:5]:
        im_array.append(Image.open('{}/train{}/{}'.format(root, year, file_name)))

    for file_name in neither_chosen[:5]:
        im_array.append(Image.open('{}/train{}/{}'.format(root, year, file_name)))
        
    fig = plt.figure(figsize=(32, 40.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 5),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

    for ax, im in zip(grid, im_array):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        
    plt.savefig('{}/examples.png'.format(name))
