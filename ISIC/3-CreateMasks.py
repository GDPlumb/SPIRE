
import joblib
import json
from multiprocessing import Pool
import os
from PIL import Image

from Config import *

def worker(i):
    filename = dataset[split][i][0]
    image = Image.open(filename)
    out = pd.get_masks(image)
    if out['pred'] == 1:
        for mode in ['pixel', 'box']:
            mask = out[mode]
            image_mask = Image.fromarray(mask).convert('RGB')
            image_mask.save('{}/{}_{}.jpg'.format(out_dir, i, mode))
    return (i, out['pred'])      
        
if __name__ == '__main__':
        
    out_dir = get_working_dir(mode = 'mask')
    os.system('rm -rf {}'.format(out_dir))
    os.system('mkdir {}'.format(out_dir))

    with open('{}/dataset.json'.format(get_working_dir()), 'r') as f:
        dataset = json.load(f)
    
    with open('./PatchDetector.joblib', 'rb') as f:
        tmp = joblib.load(f)
    pd = PatchDetector(tmp[0], tmp[1])
    
    out = {}
    for split in dataset: 
        p = Pool()
        results = p.map(worker, list(dataset[split]))
        
        out_tmp = {}
        for result in results:
            i = result[0]
            pred = result[1]
            if pred == 1:
                tmp = []
                for mode in ['pixel', 'box']:
                    tmp.append('{}/{}_{}.jpg'.format(out_dir, i, mode))
                out_tmp[i] = tmp
        
        out[split] = out_tmp
        
    with open('{}/masks.json'.format(out_dir), 'w') as f:
        json.dump(out, f)
