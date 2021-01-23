
import numpy as np
import pickle
from PIL import Image
import torch
from torchvision.datasets import VisionDataset
from torchvision import transforms
from torchvision.transforms.functional import pad

# Define the dataset and dataloader
def my_dataloader(dataset, batch_size = 64, num_workers = 3):
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

# If we have multiple versions of the same images, it is important to know that so they end up in the same fold of the dataset to prevent overfitting
def merge_sources(sources):
    file_dict = {}
    for source in sources:
        with open(source, 'rb') as f:
            data = pickle.load(f)
            
        for i in range(len(data[0])):
            filename = data[0][i]
            label = data[1][i]
            
            key = filename.split('/')[-1]
            
            if key not in file_dict:
                file_dict[key] = []
            file_dict[key].append((filename, label))
    return file_dict
    
def unpack_sources(file_dict, keys = None):

    if keys is None:
        keys = [key for key in file_dict]
    
    filenames = []
    labels = []
    
    for key in keys:
        data = file_dict[key]
        for pair in data:
            filenames.append(pair[0])
            labels.append(pair[1])

    return filenames, labels

class ImageDataset(VisionDataset):

    def __init__(self, filenames, labels, get_names = False):
        transform = get_transform()
        super(ImageDataset, self).__init__(None, None, transform, None)
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

class ImageDataset_Paired(VisionDataset):

    def __init__(self, filenames_1, labels_1, filenames_2, labels_2):
        transform = get_transform()
        super(ImageDataset_Paired, self).__init__(None, None, transform, None)
        self.filenames_1 = filenames_1
        self.labels_1 = labels_1
        self.filenames_2 = filenames_2
        self.labels_2 = labels_2
        
    def __getitem__(self, index):
        img_1 = self.transform(Image.open(self.filenames_1[index]).convert('RGB'))
        label_1 = self.labels_1[index]
        img_2 = self.transform(Image.open(self.filenames_2[index]).convert('RGB'))
        label_2 = self.labels_2[index]
        return img_1, label_1, img_2, label_2
        
    def __len__(self):
        return len(self.filenames_1)

class ImageDataset_FS(VisionDataset):

    def __init__(self, filenames, labels, contexts):
        transform = get_transform()
        super(ImageDataset_FS, self).__init__(None, None, transform, None)
        self.filenames = filenames
        self.labels = labels
        self.contexts = contexts
        
    def __getitem__(self, index):
        img = self.transform(Image.open(self.filenames[index]).convert('RGB'))
        label = self.labels[index]
        context = self.contexts[index]
        return img, label, context
        
    def __len__(self):
        return len(self.filenames)
