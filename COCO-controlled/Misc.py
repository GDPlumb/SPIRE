
import numpy as np
import sys

sys.path.insert(0, '../COCO/')
from Dataset import ImageDataset, my_dataloader
from ModelWrapper import ModelWrapper

def get_pair(coco, main, spurious):
    
    main = main.replace('+', ' ')
    spurious = spurious.replace('+', ' ')
    
    ids_main = [img['file_name'] for img in coco.get_images_with_cats([main])]
    ids_spurious = [img['file_name'] for img in coco.get_images_with_cats([spurious])]

    both = np.intersect1d(ids_main, ids_spurious)
    just_main = np.setdiff1d(ids_main, ids_spurious)
    just_spurious = np.setdiff1d(ids_spurious, ids_main)
    
    neither = [img['file_name'] for img in coco.get_images_with_cats(None)]
    neither = np.setdiff1d(neither, just_main)
    neither = np.setdiff1d(neither, just_spurious)
    
    base_dir = coco.get_base_dir()
    both = ['{}/{}'.format(base_dir, f) for f in both]
    just_main = ['{}/{}'.format(base_dir, f) for f in just_main]
    just_spurious = ['{}/{}'.format(base_dir, f) for f in just_spurious]
    neither = ['{}/{}'.format(base_dir, f) for f in neither]

    return both, just_main, just_spurious, neither
    
def process_set(model, files, label, return_value = 'acc', get_names = False, base = None):
    files_tmp = []
    labels_tmp = []
    for f in files:
        if base is None:
            files_tmp.append(f)
        else:
            files_tmp.append('{}/{}'.format(base, f))
        labels_tmp.append(np.array([label], dtype = np.float32))

    labels_tmp = np.array(labels_tmp, dtype = np.float32)

    dataset_tmp = ImageDataset(files_tmp, labels_tmp, get_names = get_names)

    dataloader_tmp = my_dataloader(dataset_tmp)
    
    model.eval()
    wrapper = ModelWrapper(model, get_names = get_names)
    if get_names:
        y_hat, y_true, names = wrapper.predict_dataset(dataloader_tmp)
    else:
        y_hat, y_true = wrapper.predict_dataset(dataloader_tmp)

    if return_value == 'acc':
        return np.mean(1 * (y_hat >= 0.5) == y_true)
    elif return_value == 'preds':
        if get_names:
            return y_hat, y_true, names
        else:
            return y_hat, y_true
            
def id_from_path(path):
    return path.split('/')[-1].split('.')[0].lstrip('0')
