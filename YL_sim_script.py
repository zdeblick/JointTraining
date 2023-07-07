array_id_str = 'SLURM_ARRAY_TASK_ID' #slurm
#array_id_str = 'PBS_ARRAYID' #pbs

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import *
from torch.utils.data import Dataset, DataLoader

os.chdir('YL_results')
id = os.getenv(array_id_str)
id = 0 if id is None else int(id)

Prange = [10,40]
Srange = [10,41]
Nrange = [10,32]
Mrange = [5,20]
Qrange = [5,20]
Pval = 2000

torch.manual_seed(id)
# trues=0
# falses=0
# for id in range(20000):
if True:
    np.random.seed(id)
    # hyperparameters of simulation
    # P = np.random.randint(Prange[0],Prange[1]+1)
    S = np.random.randint(Srange[0],Srange[1]+1)
    N = np.random.randint(Nrange[0],Nrange[1]+1)
    M = np.random.randint(Mrange[0],Mrange[1]+1)
    Q = np.random.randint(Qrange[0],Qrange[1]+1)
    P = 100
#     if N<=np.min([P,S,M+Q]):
#         trues+=1
#     else:
#         falses+=1
# print(trues,falses)
# stop
sx = 1
sy = 1
sz = 1


# True parameters of simulation
Axy = np.random.normal(size=[M,S])
Bxz = np.random.normal(size=[Q,S])
# Traing data
X = sx*np.random.normal(size=[S,P])
Y = Axy @ X + sy*np.random.normal(size=[M,P])
Z = Bxz @ X + sz*np.random.normal(size=[Q,P])
# Val data
Xv = sx*np.random.normal(size=[S,Pval])
Yv = Axy @ Xv + sy*np.random.normal(size=[M,Pval])
# val loss of simple linear regression (no z)
hAxy = Y@np.linalg.pinv(X)
Cval_ind = np.sum(np.square(Yv-hAxy @ Xv))

# val loss of Joint Training
betas = np.hstack((0,np.logspace(-5,2,15)))

class Net(nn.Module):
    '''fits Y-shaped network model to data via joint-training'''
    
    def __init__(self,S,N,M,Q):
        super(Net, self).__init__()
        self.W = nn.Linear(S, N, bias=False)
        self.A = nn.Linear(N, M, bias=False)
        self.B = nn.Linear(N, Q, bias=False)

    def forward(self, x):
        x = self.W(x)
        y = self.A(x)
        z = self.B(x)
        return z, y

Cval_JT = np.inf
epochs = 5000
lossfns = (F.mse_loss,F.mse_loss)

for beta in betas:
    model = Net(S,N,M,Q)
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    alpha = beta/(1+beta)
    for epoch in range(epochs):
        # Compute prediction error
        preds = model(torch.Tensor(X.T))
        loss = joint_loss(preds, (torch.Tensor(Z.T),torch.Tensor(Y.T)), lossfns ,alpha=alpha)
       
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    hW = model.W.weight.detach().numpy()
    hA = model.A.weight.detach().numpy()
    loss_val = np.sum(np.square(Yv-hA @ hW @ Xv))
    if loss_val<Cval_JT:
        Cval_JT = loss_val
        
print(P,S,N,M,Q)
print(Cval_ind,Cval_JT)
np.savez('YL_id='+str(id),Cval_ind=Cval_ind,Cval_JT=Cval_JT,P=P,S=S,N=N,M=M,Q=Q)






