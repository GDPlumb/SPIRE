
import sys

sys.path.insert(0, '../Common/')
from COCOWrapper import id_from_path
from Dataset import ImageDataset, my_dataloader
from LoadData import load_data

def get_map(wrapper, images, ids, name, index = 0):
    if len(ids) == 0:
        return {}
    files_tmp, labels_tmp = load_data(ids, images, [name])
    dataset_tmp = ImageDataset(files_tmp, labels_tmp, get_names = True)
    dataloader_tmp = my_dataloader(dataset_tmp)
    y_hat, y_true, names = wrapper.predict_dataset(dataloader_tmp)
    pred_map = {}
    for i in range(len(y_hat)):
        pred_map[id_from_path(names[i])] = (1 * (y_hat[i] >= 0.5))[index]
    return pred_map

# The keys in map1 need to be in map2
def get_diff(map1, map2, min_size = 25):
    n = len(map1)
    if n < min_size:
        return -1
    else:
        changed = 0
        for key in map1:
            if map1[key] != map2[key]:
                changed += 1
        return changed / n
