
import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset

# Define the dataset and dataloader
def my_dataloader(dataset, batch_size = 64, num_workers = 8):
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)

class StandardDataset(VisionDataset):

    def __init__(self, X, y):
        transform = transforms.Compose([transforms.ToTensor()])
        super(StandardDataset, self).__init__(None, None, transform, None)
        self.X = X
        self.y = y
        
    def __getitem__(self, index):
        img = self.transform(self.X[index])
        target = self.y[index]
        return img, target
        
    def __len__(self):
        return len(self.X)
        

class PairedDataset(VisionDataset):

    def __init__(self, X, y, X_prime, y_prime):
        transform = transforms.Compose([transforms.ToTensor()])
        super(PairedDataset, self).__init__(None, None, transform, None)
        self.X = X
        self.X_prime = X_prime
        self.y = y
        self.y_prime = y_prime
        
    def __getitem__(self, index):
        img = self.transform(self.X[index])
        img_prime = self.transform(self.X_prime[index])
        target = self.y[index]
        target_prime = self.y_prime[index]
        return img, target, img_prime, target_prime
        
    def __len__(self):
        return len(self.X)
