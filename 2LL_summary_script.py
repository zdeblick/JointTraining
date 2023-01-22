import numpy as np

savepath = '2LL_summaries/'

P = 50
S = 30
N = 10
Q = 8
Q_true = 8

npars = 30

prefix = '2LL_results/'
fname = 'run_A=BWp_P='+str(P)+'_S='+str(S)+'_N='+str(N)+'_Q='+str(Q)+'_Qt='+str(Q_true)
suffix = '_it='
D = np.load(prefix+fname+suffix+'0.npz',allow_pickle=True)
keys = D.keys()

big_D = {k:np.stack([ np.load(prefix+fname+suffix+str(i)+'.npz',allow_pickle=True)[k] for i in range(npars) ],axis=0) for k in keys if k not in ['pars','Wsnum','Wsanal']}
big_D['pars']=D['pars']
np.savez(fname,**big_D)
