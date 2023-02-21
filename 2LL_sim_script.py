array_id_str = 'SLURM_ARRAY_TASK_ID' #slurm
#array_id_str = 'PBS_ARRAYID' #pbs

import os
import numpy as np
from scipy.optimize import minimize
import scipy

os.chdir('2LL_results')

id = os.getenv(array_id_str)
id = 0 if id is None else int(id)

inv = np.linalg.inv
pinv = np.linalg.pinv

P = 50
S = 30
N = 10
Qs = [1,4,8]

its = 200
nv = 2000
bs = np.hstack(([0],np.logspace(-6,4,21)))

Wsnum = np.nan*np.ones((bs.size,its,N,S))
Wsanal = np.nan*np.ones((bs.size,its,N,S))


sx,sW,sy,sz,sB = 1,0.1,0.3,1,10
np.random.seed(0)
Wy = sW*np.random.randn(N,S)
Xv = sx*np.random.randn(S,nv)
xi_yv = sy*np.random.randn(N,nv)

ind_p, Qis = np.unravel_index(id,(200,len(Qs)*(len(Qs)-1) ) )
Q, Q_true = [(q,q_true) for q in Qs for q_true in Qs if q_true<=q][Qis]
np.random.seed(1+ind_p)

nullspace = scipy.linalg.null_space(Wy).T
rowspace = scipy.linalg.null_space(nullspace).T
a_r = np.random.rand(Q_true,N)
a_n = np.random.rand(Q-Q_true,S-N)
a_r2 = np.random.rand(Q,Q_true)
a_n2 = np.random.rand(Q,Q-Q_true)
a_r2/=np.sum(a_r2)
a_n2/=np.sum(a_n2)
a_b = np.hstack([a_r2,a_n2])
a_r /= np.sqrt(np.sum(np.square(a_r),axis=1,keepdims=True))
a_n /= np.sqrt(np.sum(np.square(a_n),axis=1,keepdims=True))
Bz = a_b@np.vstack([a_r@rowspace,a_n@nullspace])
Bz /= np.sqrt(np.sum(np.square(Bz),axis=1,keepdims=True))

#Bz = sB*np.random.randn(Q,S)


Enums = []
Eanals = []
Efuns = []
senums = []
seanals = []

fm = lambda w: N*np.nanmean(np.square((w-Wy)@Xv+xi_yv))
fs = lambda w: N*np.nanstd(np.nanmean(np.square((w-Wy)@Xv+xi_yv),axis=(1,2)))

A = Bz@pinv(Wy)

for it in range(its):
    np.random.seed(123456789+it)
    X = sx*np.random.randn(S,P)
    Y = Wy@X+sy*np.random.randn(N,P)
    Z = Bz@X+sz*np.random.randn(Q,P)

#     X2 = sx*np.random.randn(S,P)
#     Y2 = Wy@X2+sy*np.random.randn(N,P)
#     Z2 = Bz@X2+sz*np.random.randn(Q,P)
#     A = Z2.dot(X2.T).dot(pinv(Y2.dot(X2.T)))
#     A = Z.dot(X.T).dot(pinv(Y.dot(X.T)))


    for bi,b in enumerate(bs):

        W = inv(np.eye(N)+b*A.T.dot(A)).dot(Y+b*A.T.dot(Z)).dot(X.T).dot(inv(X.dot(X.T)))

        def func(x):
            W = x[:N*S].reshape(N,S)
            A = x[N*S:].reshape(Q,N)
            return np.sum(np.square(Y-W.dot(X)))+b*np.sum(np.square(Z-A.dot(W.dot(X))))

        def jac(x):
            W = x[:N*S].reshape(N,S)
            A = x[N*S:].reshape(Q,N)
            ddW = W.dot(X.dot(X.T))-Y.dot(X.T)+b*(A.T.dot(A).dot(W).dot(X).dot(X.T)-A.T.dot(Z).dot(X.T))
            ddA = A.dot(W.dot(X.dot(X.T)).dot(W.T))-Z.dot(X.T).dot(W.T)
            return np.hstack([2*np.ravel(ddW),2*b*np.ravel(ddA)])
        res = minimize(func,np.hstack([np.ravel(W),np.ravel(A)]),jac=jac,method='SLSQP',options={'disp':False})
        Wsanal[bi,it,:,:] = W
        W = res.x[:N*S].reshape(N,S)
        Wsnum[bi,it,:,:] = W
    
for bi,b in enumerate(bs):
    Enums.append(fm(Wsnum[bi,:,:,:]))
    senums.append(fs(Wsnum[bi,:,:,:]))
    Eanals.append(fm(Wsanal[bi,:,:,:]))
    seanals.append(fs(Wsanal[bi,:,:,:]))
    Lyb1 = sx**2*np.trace(Wy.T@Wy-2*Wy.T@inv(np.eye(N)+b*A.T@A)@(Wy+b*A.T@Bz)
                         + (Wy+b*A.T@Bz).T@inv(np.eye(N)+b*A.T@A)@inv(np.eye(N)+b*A.T@A)@(Wy+b*A.T@Bz))
    Lyb2 = S*sy**2/(P-S)*np.trace((np.eye(N)+b**2*sz**2/sy**2*A.T@A)@inv(np.eye(N)+b*A.T@A)@inv(np.eye(N)+b*A.T@A)) + N*sy**2
    Efuns.append(Lyb1+Lyb2)

by = sy**2*np.trace(A.T@A)/(3*sy**2*np.trace(A.T@A@A.T@A)+sz**2*np.trace(A.T@A)+(P-S-1)/S*sx**2*np.trace((Bz-A@Wy).T@A@A.T@(Bz-A@Wy)))
Ty = by*np.trace(A.T@A)*S/(N*(P-1))
mat = pinv(A)@Bz - Wy
Ttt = S/(P-1) * ( 1-sz**2*np.trace(inv(A@A.T))/(N*sy**2) ) - (P-S-1)/(P-1) * sx**2/(N*sy**2) * np.trace(mat.T@mat)

fname = 'run_A=BWp_P='+str(P)+'_S='+str(S)+'_N='+str(N)+'_Q='+str(Q)+'_Qt='+str(Q_true)+'_it='+str(ind_p)
np.savez(fname,Ty=Ty,by=by,Efuns=Efuns,Eanals=Eanals,Enums=Enums,seanals=seanals,senums=senums,Wsanal=Wsanal,Wsnum=Wsnum,Bz=Bz,pars=np.array((sx,sy,sz,sW,sB,P,S,N,Q,Q_true,bs,its,Wy),dtype=object))
