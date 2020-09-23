
import numpy as np
import pickle
from PIL import Image
import torch
from torchvision.datasets import VisionDataset
from torchvision import transforms
from torchvision.transforms.functional import pad

# Define the dataset and dataloader
def my_dataloader(dataset, batch_size = 64, num_workers = 4):
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, pin_memory = True)
    
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
    
class MakeSquare(object):
        
    def __call__(self, img):
        return pad(img, get_padding(img), 0, 'constant')
    
    def __repr__(self):
        return self.__class__.__name__
    
def get_transform():
    return transforms.Compose([
            MakeSquare(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

class ImageDataset(VisionDataset):

    def __init__(self, sources, get_names = False):
        transform = get_transform()
        super(ImageDataset, self).__init__(None, None, transform, None)
        
        filenames = []
        labels = []
        for source in sources:
            with open(source, 'rb') as f:
                data = pickle.load(f)
                
            for filename in data[0]:
                filenames.append(filename)
            
            for label in data[1]:
                labels.append(label)
                                                
        self.filenames = filenames
        self.labels = labels
        self.get_names = get_names
        
    def __getitem__(self, index):
        
        img = self.transform(Image.open(self.filenames[index]).convert('RGB'))
        label = self.labels[index]
        if self.get_names:
            return img, label, self.filenames[index]
        else:
            return img, label
        
    def __len__(self):
        return len(self.filenames)
