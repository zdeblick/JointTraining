import numpy as np
import os

os.chdir('YL_results')
savepath = '../YL_summaries/'

L=2
inds = 20000
eps = [1000,10000]#,5000,10000]
D0 = np.load('YL_id=0_neps=10000_L='+str(L)+'.npz')
keys = D0.keys()
big_D = {k+'s':np.zeros((inds,len(eps),D0[k].size)) for k in keys}
for id in range(inds):
    for ne, epochs in enumerate(eps):
        D = np.load('YL_id='+str(id)+'_neps='+str(epochs)+'_L='+str(L)+'.npz')
        for k in keys:
            big_D[k+'s'][id,ne,:] = D[k]

big_D = {k:np.squeeze(v) for k,v in big_D.items()}

np.savez(savepath+'YL_sum_varySNMQ_L='+str(L),**big_D)
