
import numpy as np

def id_from_path(path):
    return path.split('/')[-1].split('.')[0].lstrip('0')

def get_pair(coco, main, spurious):
    
    main = main.replace('+', ' ')
    spurious = spurious.replace('+', ' ')
    
    ids_main = [img['file_name'] for img in coco.get_images_with_cats([main])]
    ids_spurious = [img['file_name'] for img in coco.get_images_with_cats([spurious])]

    both = np.intersect1d(ids_main, ids_spurious)
    just_main = np.setdiff1d(ids_main, ids_spurious)
    just_spurious = np.setdiff1d(ids_spurious, ids_main)
    
    neither = [img['file_name'] for img in coco.get_images_with_cats(None)]
    neither = np.setdiff1d(neither, ids_main)
    neither = np.setdiff1d(neither, ids_spurious)
    
    base_dir = coco.get_base_dir()
    both = ['{}/{}'.format(base_dir, f) for f in both]
    just_main = ['{}/{}'.format(base_dir, f) for f in just_main]
    just_spurious = ['{}/{}'.format(base_dir, f) for f in just_spurious]
    neither = ['{}/{}'.format(base_dir, f) for f in neither]

    return both, just_main, just_spurious, neither
