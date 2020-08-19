
import numpy as np
import pickle
from PIL import Image
from pycocotools.coco import COCO
import torch
from torchvision.datasets import VisionDataset
from torchvision import transforms
from torchvision.transforms.functional import pad

def my_dataloader(dataset, batch_size = 32, num_workers = 4):
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)

# References
# https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/2
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/coco.py


def get_padding(image):
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding
    
def get_transform():
    return transforms.Compose([
            MakeSquare(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
class MakeSquare(object):
        
    def __call__(self, img):
        return pad(img, get_padding(img), 0, 'constant')
    
    def __repr__(self):
        return self.__class__.__name__

    
class COCODataset(VisionDataset):

    def __init__(self, root = '/home/gregory/Datasets/COCO', mode = 'val', year = '2017', sources = None, imgIds = None, get_names = False):
    
        transform = get_transform()
        super(COCODataset, self).__init__(root, None, transform, None)
                
        coco = COCO('{}/annotations/instances_{}{}.json'.format(root, mode, year))
        if imgIds is None:
            images = coco.loadImgs(coco.getImgIds())
        else:
            images = coco.loadImgs(imgIds)

        ids = []
        dim = 91 # Each 'label' vector is large enough for easy indexing, but this means it contains unused indices
        for i in range(len(images)):

            im_obj = images[i]

            filename = '{}/{}{}/{}'.format(root, mode, year, im_obj['file_name'])

            annotations = coco.loadAnns(coco.getAnnIds(im_obj['id'], iscrowd = None))
            label = np.zeros((dim), dtype = np.float32)
            for ann in annotations:
                label[ann['category_id']] = 1.0
            
            ids.append((filename, label))
            
        if sources is not None:
            for source in sources:
                with open(source, 'rb') as f:
                    info = pickle.load(f)
                for item in info:
                    ids.append(item)
            
        self.ids = ids
        self.get_names = get_names
        
    def __getitem__(self, index):

        obj = self.ids[index]
        filename = obj[0]
        target = obj[1]

        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.get_names:
            return img, target, filename
        else:
            return img, target

    def __len__(self):
        return len(self.ids)
        
class MaskedCOCODataset(VisionDataset):

    def __init__(self, source):
    
        transform = get_transform()
        
        super(MaskedCOCODataset, self).__init__(None, None, transform, None) #Do we need the 'root' from VisionDataset?
        
        with open(source, 'rb') as f:
            info = pickle.load(f)
            
        self.ids = info

        
    def __getitem__(self, index):

        obj = self.ids[index]
        filename = obj[0]
        target = obj[1]

        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.ids)

class MaskedCOCOImages(VisionDataset):

    def __init__(self, ids, coco, mask_apply = False, mask_classes = None, mask_mode = 'box', mask_unmask = True, mask_value = 'default', get_names = False):
    
        transform = get_transform()
        
        super(MaskedCOCOImages, self).__init__(None, None, transform, None)
        
        self.ids = ids
        self.coco = coco
        self.mask_apply = mask_apply
        self.mask_classes = mask_classes
        self.mask_mode = mask_mode
        self.mask_unmask = mask_unmask
        self.mask_value = mask_value
        self.get_names = get_names

    def __getitem__(self, index):
        filename = self.ids[index]

        img = Image.open(filename).convert('RGB')
        
        mask_apply = self.mask_apply
        if mask_apply:
            coco = self.coco
            mask_classes = self.mask_classes
            mask_mode = self.mask_mode
            mask_unmask = self.mask_unmask
            mask_value = self.mask_value
            
            # Get the COCO id of this image from the filename
            coco_id = np.int(filename.split('/')[-1].split('.')[0].lstrip('0'))
            
            # Get the annotations for this image
            anns = coco.loadAnns(coco.getAnnIds(imgIds = coco_id))
            
            # Calculate the mask
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
        
        img = self.transform(img)

        if self.get_names:
            return img, 0, filename
        else:
            return img, 0, target

    def __len__(self):
        return len(self.ids)
        
