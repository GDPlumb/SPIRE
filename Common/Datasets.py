
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
        


