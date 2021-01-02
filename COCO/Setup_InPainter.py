
import json
from pathlib import Path
import skimage
import sys

from Config import get_data_dir, get_data_fold

ec_source = '/home/gregory/Desktop/edge-connect'
sys.path.insert(0, ec_source)
import src.edge_connect

sys.path.insert(0, '../Common/')
from FormatData_InPainter import get_coords, get_mask, paint

if __name__ == '__main__':

    print('Painting')

    with open('./FindSCs.json', 'r') as f:
        pairs = json.load(f)
    
    ip = src.edge_connect.InPainter(model_path = '{}/checkpoints/places2/'.format(ec_source))
    
    df = get_data_fold()
    if df == -1:
        modes = ['train']
    else:
        modes = ['val']

    for mode in modes:
        mode_dir = '{}/{}'.format(get_data_dir(), mode)
        
        with open('{}/images.json'.format(mode_dir), 'r') as f:
            images = json.load(f)
        ids = [id for id in images]
        
        for pair in pairs:
            main = pair.split('-')[0]
            spurious = pair.split('-')[1]
            pair_dir = '{}/{}-{}'.format(mode_dir, main, spurious)
            print(pair_dir)
            
            for object in [main, spurious]:
                mask_mode = '{}-{}-pixel'.format(main, object)
                save_dir = '{}/{}-paint'.format(pair_dir, mask_mode)
                Path(save_dir).mkdir(parents = True, exist_ok = True)
        
                for id in ids:
                    if mask_mode in images[id]:
                        file = images[id][mask_mode][0]
                        label = images[id][mask_mode][1].copy()
                        
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
                        file_new = file.split('/')[-1]
                        file_new = file_new[:-3] + 'jpg'
                        file_new = '{}/{}'.format(save_dir, file_new)
                        
                        skimage.io.imsave(file_new, a_rs)
                        
                        images[id]['{}-paint'.format(mask_mode)] = [file_new, label]

            with open('{}/images.json'.format(mode_dir), 'w') as f:
                json.dump(images, f)
