
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
        
        for pair in pairs:
            main = pair.split('-')[0]
            spurious = pair.split('-')[1]
            pair_dir = '{}/{}-{}'.format(mode_dir, main, spurious)
            print(pair_dir)
            
            for object in ['main', 'spurious']:
                # Get the relevant images
                with open('{}/both-{}-pixel/images.json'.format(pair_dir, object), 'r') as f:
                    images = json.load(f)
                
                # Setup the output directory
                save_dir = '{}/both-{}-pixel-paint'.format(pair_dir, object)
                Path(save_dir).mkdir(parents = True, exist_ok = True)
        
                # In-Paint the images
                images_new = {}
                for id in images:
                    file = images[id][0]
                    label = images[id][1].copy()
                    
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
                    
                    images_new[id] = [file_new, label]

                # Save the output
                with open('{}/images.json'.format(save_dir), 'w') as f:
                    json.dump(images_new, f)
