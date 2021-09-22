
import torch

#c != 0 -> supress the context and weight by c
#c = 0 -> do not supress and weight by 1
def fs_loss(rep, rep_avg_running, model, metric_loss, y, c):
    rep = torch.squeeze(rep)
    if len(rep.shape) == 1: #Fix the shape for a batch_size of 1
        rep = torch.unsqueeze(rep, 0)
        
    batch_size = rep.shape[0]
    dim = rep.shape[1]
    priv_len = int(dim / 2)

    # Keep an exponential average of the represntation
    if rep_avg_running is None:
        rep_avg_running = torch.mean(rep, dim = 0).detach()
    else:
        rep_avg_running = 0.8 * rep_avg_running + 0.2 * torch.mean(rep, dim = 0).detach()
    
    # Split the features into those that are "exclusive to the object" (eg, for the object without common context) and those that are not
    binary_wt_exclusive = torch.ones(batch_size, dim).cuda()
    for b_im_it in range(batch_size):
        binary_wt_exclusive[b_im_it][0:priv_len] = 1
        binary_wt_exclusive[b_im_it][priv_len:dim] = 0
    binary_wt_non_exclusive = torch.ones(batch_size, dim).cuda()
    
    # Construct the representation for each of those settings
    rep_exclusive = torch.mul(rep, binary_wt_exclusive)
    for b_im_it in range(batch_size):
        rep_exclusive[b_im_it][priv_len:dim] = rep_avg_running[priv_len:dim]
    rep_non_exclusive = torch.mul(rep, binary_wt_non_exclusive)
    
    # Get the model's output for each of those settings
    # WARNING:  assumes that the next layer is the linear classification layer
    out_exclusive = model.fc(rep_exclusive)
    out_non_exclusive = model.fc(rep_non_exclusive)
    
    # Calculate the loss
    loss_exclusive = metric_loss(out_exclusive, y)
    loss_exclusive = torch.mul(loss_exclusive, c)

    loss_non_exclusive = metric_loss(out_non_exclusive, y)
    loss_non_exclusive = torch.mul(loss_non_exclusive, 1.0 * (c == 0.0))

    loss = loss_exclusive + loss_non_exclusive
    loss = torch.mean(loss)
    
    return loss, rep_avg_running
