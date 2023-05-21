import numpy as np
inv = np.linalg.inv
pinv = np.linalg.pinv

savepath = '2LL_summaries/'

npars = 30
prefix = '2LL_results/'
suffix = '_it='


P = 50
S = 30
N = 10
Qs = [1,4,8]
for Qis in range(len(Qs)*(len(Qs)-1)):
    Q, Q_true = [(q,q_true) for q in Qs for q_true in Qs if q_true<=q][Qis]

    fname = 'run_A=BWp_P='+str(P)+'_S='+str(S)+'_N='+str(N)+'_Q='+str(Q)+'_Qt='+str(Q_true)

    D = np.load(prefix+fname+suffix+'0.npz',allow_pickle=True)
    keys = D.keys()
    big_D = {k:np.stack([ np.load(prefix+fname+suffix+str(i)+'.npz',allow_pickle=True)[k] for i in range(npars) ],axis=0) for k in keys if k not in ['pars','Wsnum','Wsanal']}
    big_D['pars']=D['pars']
    np.savez(savepath+fname,**big_D)
