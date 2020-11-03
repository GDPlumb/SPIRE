
import os
from pathlib import Path
import pickle
import skimage
import sys

from Config import get_data_dir

ec_source = '/home/gregory/Desktop/edge-connect'
sys.path.insert(0, ec_source)
import src.edge_connect

sys.path.insert(0, '../COCO/')
from FormatData_InPainter import get_coords, get_mask, paint

if __name__ == '__main__':

    main = sys.argv[1]
    spurious = sys.argv[2]
    
    ip = src.edge_connect.InPainter(model_path = '{}/checkpoints/places2/'.format(ec_source))

    pair_dir = '{}/{}-{}'.format(get_data_dir(), main, spurious)
    for mode in ['val', 'train']:
        mode_dir = '{}/{}'.format(pair_dir, mode)
        
        with open('{}/images.p'.format(mode_dir), 'rb') as f:
            images = pickle.load(f)
        ids = [id for id in images]
        
        
        for mask_mode in ['box-main', 'box-spurious', 'pixel-main', 'pixel-spurious']:
            for id in ids:
                if mask_mode in images[id]:
                    file = images[id][mask_mode][0]
                    label = images[id][mask_mode][1]
                    
                    # Load the image
                    image = skimage.io.imread(file)
                    width, height, _  = image.shape

                    # Get the mask
                    coords = get_coords(image)
                    mask = get_mask(coords, width, height)
                
                    # Format the image and mask for the inpainter
                    image_rs = skimage.util.img_as_ubyte(skimage.transform.resize(image, (256, 256)))
                    mask_rs = skimage.util.img_as_ubyte(skimage.transform.resize(mask, (256, 256)))

                    # Inpaint the image
                    a = paint(ip, image_rs, mask_rs)
                    a_rs = skimage.transform.resize(a.astype('uint8'), (width, height))
                    
                    # Save the inpainted image
                    index = file.rfind('/')
                    file_new = file[:index] + '-paint' + file[index:]
                    file_new = file_new[:-3] + 'jpg'
                    Path(file_new[:file_new.rfind('/')]).mkdir(parents = True, exist_ok = True)
                    skimage.io.imsave(file_new, a_rs)
                    
                    images[id]['{}-paint'.format(mask_mode)] = [file_new, label]

        with open('{}/images.p'.format(mode_dir), 'wb') as f:
            pickle.dump(images, f)
            
        
