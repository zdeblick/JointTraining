import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


def joint_loss(pred,y,lossfns,alpha=1.0):
    '''
    cost function used for joint-training
    Args:
        pred: length-2 list of predicted model ouputs for hypothesized task and neural responses
        y: length-2 list of target ouputs for hypothesized task and neural responses
        lossfns: length-2 list of pytorch loss functions for hypothesized task and neural responses
        alpha: hyperparameter weighting these two cost functions (alpha = beta/(1+beta))
    Returns:
        joint cost function (pytorch loss function)
    '''
    return alpha*lossfns[0](pred[0],y[0])+(1-alpha)*lossfns[1](pred[1],y[1])



def dim(x):
    if type(x)==torch.Tensor:
        x = x.detach().numpy()
    x = x-np.mean(x,axis=0)
    C = x.T@x
    eigs,_ = np.linalg.eig(C)
    return np.square(np.sum(eigs))/np.sum(np.square(eigs))
