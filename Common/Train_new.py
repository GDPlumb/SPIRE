
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch

def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda()
    x_mixed = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return x_mixed, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
def train_model(model, dataloaders, criterion, optimizer, lr,
                lr_decay_rate = 0.5, lr_decay_delay = 4, lr_decay_count = 5, name = 'history'):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    best_epoch = 0
    
    history = {}
    history['train'] = []
    history['val'] = []
    history['rate'] = []
    
    epoch = -1
    decays = 0
    while True:
    
        # Check convergence and update the learning rate accordingly
        epoch += 1
        if epoch - best_epoch > lr_decay_delay:
            decays += 1
            if decays > lr_decay_count:
                model.load_state_dict(best_model_wts)
                return model
            else:
                history['rate'].append(epoch)
                lr = lr * lr_decay_rate
                best_epoch = epoch
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            counts_loss = 0
            # Iterate over data.
            for data in dataloaders[phase]:

                x = data[0]
                y = data[1]

                x = x.to('cuda')
                batch_size = x.size(0)
                y = y.to('cuda')
                                        
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    pred = model(x)
                    loss_main = criterion(pred, y)
                     
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss = loss_main
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * batch_size
                counts_loss += batch_size

            epoch_loss = running_loss / counts_loss
            history[phase].append(epoch_loss)

            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        x = [i for i in range(epoch + 1)]
        plt.scatter(x, history['train'], label = 'train')
        plt.scatter(x, history['val'], label = 'val')
        for e in history['rate']:
            plt.axvline(e, color = 'black', linestyle = '--')

        plt.legend()
        plt.savefig('{}.pdf'.format(name))
        plt.close()
