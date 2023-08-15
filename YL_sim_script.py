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

Ls = [1,2,4]
Prange = [35,50]
Srange = [10,41]
Nrange = [10,32]#[14,35]
Mrange = [5,20]
Qrange = [5,20]
Pval = 2000
epochs = 20000
trials = 325

Li,seed = np.unravel_index(id,(len(Ls),trials))
L = Ls[Li]
torch.manual_seed(seed)
trues=0
falses=0
#for seed in range(20000):
if True:
    np.random.seed(seed)
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
    if id<10:
        print(Ns)
print(trues,falses)
#stop
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

# val loss of Joint Training
betas = np.hstack((0,np.logspace(-5,2,15)))
bi_chosen = 7 #1e-2

class Net(nn.Module):
    '''fits Y-shaped network model to data via joint-training'''
    
    def __init__(self,S,Ns,M,Q):
        if type(Ns) is int:
            Ns = [int(Ns)]
        super(Net, self).__init__()
        self.W0 = nn.Linear(S, Ns[0], bias=False)
        for i in range(len(Ns)-1):
            setattr(self,'W'+str(i+1),nn.Linear(Ns[i], Ns[i+1], bias=False) )
        self.A = nn.Linear(Ns[-1], M, bias=False)
        self.B = nn.Linear(Ns[-1], Q, bias=False)
        self.L = len(Ns)

    def forward(self, x):
        for i in range(self.L):
            x = getattr(self,'W'+str(i))(x)
        y = self.A(x)
        z = self.B(x)
        return z, y

    def dims(self, x):
        with torch.no_grad():
            dims = [dim(x)]
            for i in range(self.L):
                x = getattr(self,'W'+str(i))(x)
                dims.append(dim(x))
            y = self.A(x)
            dims.append(dim(y))
            z = self.B(x)
            dims.append(dim(z))
        return dims

    def m_dim(self):
        with torch.no_grad():
            M = self.W0.weight.data
            for i in range(1,self.L):
                M = getattr(self,'W'+str(i)).weight.data @ M
        return erank(M)

lossfns = (F.mse_loss,F.mse_loss)
Cvals = np.zeros((betas.size,epochs//100))
dims = np.zeros((betas.size,L+3,epochs//100))
eranks = np.zeros((betas.size,epochs//100))

for bi,beta in enumerate(betas):
    model = Net(S,Ns,M,Q)
    model = model.float()
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    alpha = beta/(1+beta)
    for epoch in range(epochs):
        # Compute prediction error
        preds = model(torch.Tensor(X.T))
        loss = joint_loss(preds, (torch.Tensor(Z.T),torch.Tensor(Y.T)), lossfns ,alpha=alpha)
       
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#        if epoch%(epochs//10)==(epochs//10)-1:
            #lr*=0.5
#            optimizer.param_groups[0]['lr'] = lr
#            print(loss.item())
        if epoch%100==99:
            loss_val = F.mse_loss(model(torch.Tensor(Xv.T))[1],torch.Tensor(Yv.T)).detach().numpy()
            Cvals[bi,epoch//100] = loss_val
            dims[bi,:,epoch//100] = model.dims(torch.Tensor(X.T))
            eranks[bi,epoch//100] = model.m_dim()

Cval_JT = np.min(Cvals,axis=0)
Cval_ind = Cvals[0,:]
bstari = np.argmin(Cvals,axis=0)
dims_1b = dims[bi_chosen,:,:]
dims = dims[bstari,:,np.arange(epochs//100)]        
eranks_1b = eranks[bi_chosen,:]
eranks = eranks[bstari,np.arange(epochs//100)]
out_dims = [dim(Y.T), dim(Z.T)]

print(P,S,Ns,M,Q)
print(Cval_ind,Cval_JT)
np.savez('YL_id='+str(seed)+'_neps='+str(epochs)+'_L='+str(L),Cval_ind=Cval_ind,Cval_JT=Cval_JT,P=P,S=S,N=Ns,M=M,Q=Q,dims=dims,out_dims=out_dims,bstari=bstari,dims_1b=dims_1b,eranks_1b=eranks_1b,eranks=eranks,b=betas[bi_chosen],betas=betas)






