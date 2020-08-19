
import numpy as np
from PIL import Image

from COCODataset import get_transform

class ImageLoader():

    def __init__(self, root, coco):
        self.root = root
        self.coco = coco
        self.transform = get_transform()

    def load_img(self, img_obj, transform_apply = True, mask_apply = False, mask_classes = None, mask_mode = 'box', mask_unmask = True, mask_value = 'default'):
        coco = self.coco

        img = Image.open('{}{}'.format(self.root, img_obj['file_name'])).convert('RGB')
        
        if mask_apply:
            anns = coco.loadAnns(coco.getAnnIds(imgIds = img_obj['id']))
            
            mask_classes = coco.getCatIds(catNms = mask_classes)

            mask = []
            for ann in anns:
                if ann['category_id'] in mask_classes:
                    tmp = coco.annToMask(ann)
                    if mask_mode == 'pixel':
                        mask.append(tmp)
                    elif mask_mode == 'box':
                        
                        idx = np.where(tmp == 1.0)
                        if len(idx[0] > 0): #BUG?  Sometimes this has length 0 and things break
                            min_0 = np.min(idx[0])
                            max_0 = np.max(idx[0])
                            min_1 = np.min(idx[1])
                            max_1 = np.max(idx[1])
                            
                            tmp_new = np.copy(tmp)
                            tmp_new[min_0:max_0, min_1:max_1] = 1.0
                            
                            mask.append(tmp_new)
                        else:
                            mask.append(tmp)
                            
            if len(mask) > 0:
                mask = np.expand_dims(1.0 * (np.sum(np.array(mask), axis = 0) >= 1.0), axis = 2)
                
                if mask_unmask:
                    unmask = []
                    for ann in anns:
                        if ann['category_id'] not in mask_classes:
                            tmp = coco.annToMask(ann)
                            unmask.append(tmp)

                    if len(unmask) > 0:
                        unmask = np.expand_dims(1.0 * (np.sum(np.array(unmask), axis = 0) >= 1.0), axis = 2)
                        mask = mask - unmask
                        mask = np.clip(mask, 0, 1)
                
                mask = (np.squeeze(mask) == 1)
                img_np = np.array(img)
                if mask_value == 'default':
                    img_np[mask] = [124, 116, 104]
                elif mask_value == 'random':
                    img_np[mask] =  np.random.randint(low = 0, high = 256, size = (np.sum(mask), 3))
                elif mask_value == 'mean':
                    img_np[mask] = np.mean(np.array(img), axis = (0,1)).astype(np.int)
                img = Image.fromarray(img_np)
                            
        if transform_apply:
            return self.transform(img)
        else:
            return img

    def t2np(self, img):
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        return img
