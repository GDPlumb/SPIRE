
import numpy as np
import skimage
import sys

SOURCE = '../../edge-connect/'
sys.path.insert(0, SOURCE)
import src.edge_connect

# https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

class InPaintWrapper():
    
    def __init__(self, source = SOURCE):
        self.source = source
        self.painter = src.edge_connect.InPainter(model_path = '{}/checkpoints/places2/'.format(source))
        
    def get_mask(self, image, color = [124, 116, 104], tol = 0):
        width, height, _  = image.shape
        dif = np.abs(image - color)
        dif = np.mean(dif, axis= 2)
        coords =  np.where(dif <= tol)
        mask = np.zeros((width, height))
        mask[coords[0], coords[1]] = 1
        return mask
           
    def inpaint(self, filenames, batch_size = 32):
        # Setup the images to be inpainted
        images = []
        masks = []
        sizes = []
        for filename in filenames:
            image = skimage.io.imread(filename)
            mask = self.get_mask(image)
            
            image_rs = skimage.util.img_as_ubyte(skimage.transform.resize(image, (256, 256)))
            mask_rs = skimage.util.img_as_ubyte(skimage.transform.resize(mask, (256, 256)))
            
            images.append(image_rs)
            masks.append(mask_rs)
            sizes.append(mask.shape)
            
        # Inpaint the images
        images_batch = batch(images, n = batch_size)
        masks_batch = batch(masks, n = batch_size)
        images_painted = []
        while True:
            try:
                image_batch = next(images_batch)
                mask_batch = next(masks_batch)
                out = self.painter.inpaint(image_batch, mask_batch)
                for v in out:
                    images_painted.append(v.cpu().numpy())
            except StopIteration:
                break
        
        # Restore the images to their original size
        for i, v in enumerate(sizes):
            images_painted[i] = skimage.transform.resize(images_painted[i].astype('uint8'), v)
            
        return images_painted