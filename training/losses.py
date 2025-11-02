import torch.nn.functional as F

def cross_entropy_loss(pred,target):
    return F.cross_entropy(pred,target)

def mse(pred,target):
    return F.mse_loss(pred,target)