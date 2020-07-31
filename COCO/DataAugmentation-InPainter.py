

import numpy as np
import os
from pathlib import Path
import pickle
import skimage
import sys
from tqdm import tqdm

def get_coords(image, color = [124, 116, 104], tol = 0):
    dif = np.abs(image - color)
    dif = np.mean(dif, axis= 2)
    out =  np.where(dif <= tol)
    return out
    
def get_mask(coords, width, height):
    out = np.zeros((width, height))
    out[coords[0], coords[1]] = 1
    return out
    
def paint(ip, image, mask):
    return ip.inpaint([image], [mask])[0].cpu().numpy()
    
ec_source = '/home/gregory/Desktop/edge-connect'
sys.path.insert(0, ec_source)
import src.edge_connect
ip = src.edge_connect.InPainter(model_path = '{}/checkpoints/places2/'.format(ec_source))

for mode in ['val', 'train']:
    for shape in ['box', 'pixel']:
    
        if mode == 'train' and shape == 'pixel':
            break
        
        dataset = '{}2017-none-[person]'.format(mode)
        mask_config = '{}-True-default'.format(shape)
        save_location = './DataAugmentation/{}/{}-True-paint/'.format(dataset, shape)
        
        print('')
        print(save_location)
        print('')
    
        os.system('rm -rf {}'.format(save_location))
        Path(save_location).mkdir(parents=True, exist_ok=True)

        with open('./DataAugmentation/{}/{}/labels.p'.format(dataset, mask_config), 'rb') as f:
            info = pickle.load(f)
    
        #import cProfile
        #pr = cProfile.Profile()
        #pr.enable()
        
        info_new = []
        for i in tqdm(range(len(info))):
            
            #fname = '{}png'.format(info[i][0][:-3])
            
            image = skimage.io.imread(info[i][0])
            width, height, _  = image.shape

            coords = get_coords(image)
            
            mask = get_mask(coords, width, height)

            image_rs = skimage.util.img_as_ubyte(skimage.transform.resize(image, (256, 256)))
            mask_rs = skimage.util.img_as_ubyte(skimage.transform.resize(mask, (256, 256)))

            a = paint(ip, image_rs, mask_rs)
            
            a_rs = skimage.transform.resize(a.astype('uint8'), (width, height))
            
            fname = '{}{}'.format(save_location, info[i][0].split('/')[-1])
            
            skimage.io.imsave(fname, a_rs)
            
            info_new.append((fname, info[i][1]))


        with open('{}/labels.p'.format(save_location), 'wb') as f:
            pickle.dump(info_new, f)

        
        #pr.disable()
        #pr.print_stats(sort='time')
