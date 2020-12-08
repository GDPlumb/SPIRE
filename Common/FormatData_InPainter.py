
import numpy as np
import os
import pickle
import skimage
import sys

ec_source = '/home/gregory/Desktop/edge-connect'
sys.path.insert(0, ec_source)
import src.edge_connect


## This is to preven a decompression error
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True

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
    
if __name__ == "__main__":

    root = sys.argv[1]
    mode = sys.argv[2]
    year = sys.argv[3]
    task = sys.argv[4]
    
    base_location = '{}/{}{}-{}'.format(root, mode, year, task)
    
    with open('{}-pixel-info.p'.format(base_location), 'rb') as f:
        data = pickle.load(f)
    filenames = data[0]
    labels = data[1]
    
    save_location = '{}-paint'.format(base_location)
    os.system('rm -rf {}'.format(save_location))
    os.system('mkdir {}'.format(save_location))
    
    ip = src.edge_connect.InPainter(model_path = '{}/checkpoints/places2/'.format(ec_source))

    filenames_new = []
    for filename in filenames:
        
        # Load the image
        image = skimage.io.imread(filename)
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
    
        # Get the new filename
        filename_new = '{}/{}.jpg'.format(save_location, filename.split('/')[-1].split('.')[0])
        skimage.io.imsave(filename_new, a_rs)
        filenames_new.append(filename_new)
        
    with open('{}-info.p'.format(save_location), 'wb') as f:
        pickle.dump([filenames_new, labels], f)
    
    model_class = sys.argv[1].replace('-', ' ')
    labeler_class = sys.argv[2].replace('-', ' ')
    year = sys.argv[3]
    ec_source = sys.argv[4]
