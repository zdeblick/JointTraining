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

L = 4
Prange = [50,50]#[35,50]
Srange = [10,41]
Nrange = [14,35]
Mrange = [5,20]
Qrange = [5,20]
Pval = 2000
epochs = 200


trues=0
falses=0
#for id in range(20000):
if True:
    np.random.seed(id)
    # hyperparameters of simulation
    P = np.random.randint(Prange[0],Prange[1]+1)
    S = np.random.randint(Srange[0],Srange[1]+1)
    Ns = [np.random.randint(Nrange[0],Nrange[1]+1) for i in range(L)]
    M = np.random.randint(Mrange[0],Mrange[1]+1)
    Q = np.random.randint(Qrange[0],Qrange[1]+1)
    if np.min(Ns)<np.min([P,S,M+Q]):
        trues+=1
    else:
        falses+=1
print(trues,falses)
#stop
sx = 1
sy = 1
sz = 1
#S,Ns,M,Q = (20,[22]*L,10,10)
#S,Ns,M,Q = (40,[15]*L,20,20)

# True parameters of simulation
Axy = np.random.normal(size=[M,S])
#Bxz = Axy
Bxz = np.random.normal(size=[Q,S])
# Traing data
X = sx*np.random.normal(size=[S,P])
Y = Axy @ X + sy*np.random.normal(size=[M,P])
Z = Bxz @ X + sz*np.random.normal(size=[Q,P])
# Val data
Xv = sx*np.random.normal(size=[S,Pval])
Yv = Axy @ Xv + sy*np.random.normal(size=[M,Pval])


class Net(nn.Module):
    '''fits Y-shaped network model to data via joint-training'''
    
    def __init__(self,S,Ns,M,Q,mats=None):
        if type(Ns) is int:
            Ns = [int(Ns)]
        super(Net, self).__init__()
        self.W = [nn.Linear(S, Ns[0], bias=False,)]
        for i in range(len(Ns)-1):
            self.W.append(nn.Linear(Ns[i], Ns[i+1], bias=False))
        self.A = nn.Linear(Ns[-1], M, bias=False)
        self.B = nn.Linear(Ns[-1], Q, bias=False)
        if mats is not None:
            for i,mat in enumerate(mats[:-1]):
                self.W[i].weight.data = torch.Tensor(mat)
            self.A.weight.data = torch.Tensor(mats[-1])

    def forward(self, x):
        for Wi in self.W:
            x = Wi(x)
        y = self.A(x)
        z = self.B(x)
        return z, y


lossfns = (F.mse_loss,F.mse_loss)
# val loss of Joint Training
betas = np.hstack((0,np.logspace(-6,6,13),np.inf))
Cvals = np.zeros((betas.size,epochs//1))

for bi,beta in enumerate(betas):
    torch.manual_seed(id)

    alpha = beta/(1+beta) if np.isfinite(beta) else 1

    model = Net(S,Ns,M,Q)
    model = model.float()
    optimizer = torch.optim.LBFGS(model.parameters())

    for epoch in range(epochs):
     
        # Backpropagation
        def closure():
#        if True:
            # Compute prediction error
            optimizer.zero_grad()
            preds = model(torch.Tensor(X.T))
            loss = joint_loss(preds, (torch.Tensor(Z.T),torch.Tensor(Y.T)), lossfns ,alpha=alpha)
            loss.backward()
            return loss
        optimizer.step(closure)
        if True or epoch%10==9:
            loss_val = float(F.mse_loss(model(torch.Tensor(Xv.T))[1],torch.Tensor(Yv.T)).detach().numpy())
            Cvals[bi,epoch//1] = loss_val
            print('beta=',beta, 'vloss', loss_val)

print(P,S,Ns,M,Q)
np.savez('YL_LBFGS_id='+str(id)+'_neps='+str(epochs)+'_L='+str(L),Cvals=Cvals,P=P,S=S,N=Ns,M=M,Q=Q)






