
import copy
import numpy as np
import time
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
    
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs = 5,
                mixup_weight = None, mixup_alpha = 0.1,
                rrr_weight = None,
                verbose = True):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    for epoch in range(num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

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
                                
                if rrr_weight is not None:
                    x_prime = data[2]
                    x_prime = x_prime.to('cuda')

                    x.requires_grad = True
                                        
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train' or rrr_weight is not None):
                    pred = model(x)
                    loss_main = criterion(pred, y)

                    if mixup_weight is not None:
                        x_mixed, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)
                        pred_mixed = model(x_mixed)
                        loss_mixed = mixup_criterion(criterion, pred, y_a, y_b, lam)
   
                    if rrr_weight is not None:
                        diff = torch.max(1.0 * (x != x_prime), 1, keepdim = True, out = None)[0]
                        
                        prob = torch.sigmoid(pred)
                        grad = torch.autograd.grad(prob, x, grad_outputs = torch.ones(prob.shape).to('cuda'), create_graph = True)[0]
                        loss_rrr = torch.norm(diff * grad, 2)
                     
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if mixup_weight is not None:
                            loss = loss_main + mixup_weight * loss_mixed
                        elif rrr_weight is not None:
                            loss = loss_main + rrr_weight * loss_rrr
                        else:
                            loss = loss_main
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * batch_size
                counts_loss += batch_size
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / counts_loss

            if verbose:
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        if verbose:
            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
