
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Define a simple ConvNet
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = (1,1))
        self.conv2 = nn.Conv2d(32, 32, kernel_size = 3, padding = (1,1))
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, padding = (1,1))
        self.conv4 = nn.Conv2d(64, 64, kernel_size = 3, padding = (1,1))
        self.fc1 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def accuracy(self, dataloader):
        correct = 0
        total = 0
        for x, y in dataloader:
            pred = 1.0 * (self(x.cuda()).cpu().data.numpy() > 0)
            true = y.data.numpy()
            correct += np.sum(pred == true)
            total += pred.shape[0]
        return correct / total
