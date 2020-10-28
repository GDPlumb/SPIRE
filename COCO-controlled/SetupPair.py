
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
    
    num_samples = 1000

    # Get the image splits for the training data
    coco = COCOWrapper(mode = 'train')
    both, just_main, just_spurious, neither = get_pair(coco, main, spurious)
    
    # Randomly chose num_sample from each of the splits to use for training
    both_chosen = random.sample(list(both), num_samples)
    just_main_chosen = random.sample(list(just_main), num_samples)
    just_spurious_chosen = random.sample(list(just_spurious), num_samples)
    neither_chosen = random.sample(list(neither), num_samples)
    
    # Setup the directory and save the output
    name = './Pairs/{}-{}'.format(main, spurious)
    os.system('rm -rf {}'.format(name))
    os.system('mkdir {}'.format(name))
    
    with open('{}/splits.p'.format(name), 'wb') as f:
        pickle.dump([both_chosen, just_main_chosen, just_spurious_chosen, neither_chosen], f)
    
    # Check the splits look right
    im_array = []
    for f in both_chosen[:5]:
        im_array.append(Image.open(f))
        
    for f in just_main_chosen[:5]:
        im_array.append(Image.open(f))

    for f in just_spurious_chosen[:5]:
        im_array.append(Image.open(f))

    for f in neither_chosen[:5]:
        im_array.append(Image.open(f))
        
    fig = plt.figure(figsize=(40, 50))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 5),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

    for ax, im in zip(grid, im_array):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        
    plt.savefig('{}/Examples.png'.format(name))
