
import numpy as np
import pickle
from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset
from torchvision import transforms
from torchvision.transforms.functional import pad


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

    def __init__(self, root = '/home/gregory/Datasets/COCO/', mode = 'val', year = '2017'):
    
        transform = get_transform()
        super(COCODataset, self).__init__(root, None, transform, None)
                
        coco = COCO('{}/annotations/instances_{}{}.json'.format(root, mode, year))
        images = coco.loadImgs(coco.getImgIds())

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
            
        self.ids = ids

        
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
