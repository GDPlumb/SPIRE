
import numpy as np

def get_pair(coco, main, spurious):
    ids_main = [img['file_name'] for img in coco.get_images_with_cats([main])]
    ids_spurious = [img['file_name'] for img in coco.get_images_with_cats([spurious])]

    both = np.intersect1d(ids_main, ids_spurious)
    just_main = np.setdiff1d(ids_main, ids_spurious)
    just_spurious = np.setdiff1d(ids_spurious, ids_main)
    
    neither = [img['file_name'] for img in coco.get_images_with_cats(None)]
    neither = np.setdiff1d(neither, just_main)
    neither = np.setdiff1d(neither, just_spurious)

    return both, just_main, just_spurious, neither
