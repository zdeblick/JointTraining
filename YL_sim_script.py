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
id = 5 if id is None else int(id)

L = 1
Prange = [50,50]#[35,50]
Srange = [10,41]
Nrange = [14,35]
Mrange = [5,20]
Qrange = [5,20]
Pval = 2000
epochs = 2000


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
S,Ns,M,Q = (40,[15]*L,20,20)

# True parameters of simulation
Axy = np.random.normal(size=[M,S])
Bxz = Axy
#Bxz = np.random.normal(size=[Q,S])
# Traing data
X = sx*np.random.normal(size=[S,P])
Y = Axy @ X + sy*np.random.normal(size=[M,P])
Z = Bxz @ X + sz*np.random.normal(size=[Q,P])
# Val data
Xv = sx*np.random.normal(size=[S,Pval])
Yv = Axy @ Xv + sy*np.random.normal(size=[M,Pval])

hAxy = Y@np.linalg.pinv(X)
UU,SS,VVh = np.linalg.svd(hAxy,full_matrices=False)
r = np.min([np.min(Ns),S,M])
hAxy = UU[:,:r]@np.diag(SS[:r])@VVh[:r,:]
print(np.mean(np.square(Yv-hAxy @ Xv)), np.mean(np.square(Y-hAxy @ X)),)
mats = [ np.vstack([ np.diag(SS[:r])@VVh[:r,:],np.zeros((Ns[0]-r,S)) ]) ]
for i in range(L-1):
    mat = np.zeros((Ns[i+1],Ns[i]))
    mat[np.arange(r),np.arange(r)] = 1
    mats.append(mat)
mats.append( np.hstack([ UU[:,:r],np.zeros((M,Ns[-1]-r)) ]) )
for mat in mats:
    mat += 0.2/(L+1)*np.random.normal(size=mat.shape)


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


Cvals = []
lossfns = (F.mse_loss,F.mse_loss)
# val loss of Joint Training
#betas = np.hstack((0,np.logspace(-5,4,10)))
betas = np.array([0,1e-3])


for beta in betas:
    alpha = beta/(1+beta)
    torch.manual_seed(id)

    model = Net(S,Ns,M,Q)#,mats)
    model = model.float()
    all_params = set(model.parameters())
    task_head_params = set(model.B.parameters())
    act_head_params = set(model.A.parameters())
    base_params = all_params-task_head_params-act_head_params
    base_params = list(base_params)
    task_head_params = list(task_head_params)
    act_head_params = list(act_head_params)    
    lr = 1e-2
    optimizer = torch.optim.Adam([{'params':base_params},{'params':task_head_params,'lr':lr/(alpha+1e-10)},{'params':act_head_params,'lr':lr/(1-alpha+1e-10)}], lr=lr)
#    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True,factor=0.3,patience=1,cooldown=5,min_lr = 3e-3,threshold=-1e-6)
    old_loss = 10000000

    for epoch in range(epochs):
        # Compute prediction error
        preds = model(torch.Tensor(X.T))
        loss = joint_loss(preds, (torch.Tensor(Z.T),torch.Tensor(Y.T)), lossfns ,alpha=alpha)
     
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 #       scheduler.step(loss)
        new_loss = loss.item()
        old_loss = new_loss
        if epoch%(epochs/10)==0:
            lr*=0.9
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr/(alpha+1e-10)
            optimizer.param_groups[2]['lr'] = lr/(1-alpha+1e-10)
            print(new_loss)
    loss_val = float(F.mse_loss(model(torch.Tensor(Xv.T))[1],torch.Tensor(Yv.T)).detach().numpy())
    print('beta=',beta, 'vloss', loss_val)
    Cvals.append(loss_val)
    if beta==0:
        Cval_ind = loss_val

print(P,S,Ns,M,Q)
dCvdb0 = (Cvals[1]-Cvals[0])/(betas[1])
Cval_JT = np.min(Cvals)
print(Cval_ind,Cval_JT,dCvdb0,(Cval_ind-Cval_JT)/Cval_ind)
#np.savez('YLdb_id='+str(id)+'_neps='+str(epochs)+'_L='+str(L),Cval_ind=Cval_ind,Cval_JT=Cval_JT,P=P,S=S,N=Ns,M=M,Q=Q,dCvdb0=dCvdb0)






