import numpy as np
import torch
import torch.nn.functional as F

def compute_accuracy(y_pred, y_target):
    y_target = y_target.cpu()
    _, y_pred_indices = y_pred.cpu().max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

def compute_perplexity(y_pred, y_true, apply_softmax=False):
    if apply_softmax:
        y_pred = F.softmax(y_pred, dim=1)
        
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    indices = np.arange(len(y_true))
    
    return 2**np.mean(-np.log2(y_pred[indices, y_true]))