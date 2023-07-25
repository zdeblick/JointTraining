import numpy as np
import os

os.chdir('YL_results')
savepath = '../YL_summaries/'

L=4
inds = 324
eps = [1000,5000,10000]
D0 = np.load('YL_id=0_neps='+str(eps[-1])+'_L='+str(L)+'.npz')
keys = D0.keys()
big_D = {k+'s':np.nan*np.ones((inds,len(eps),D0[k].size)) for k in keys}
for id in range(inds):
    for ne, epochs in enumerate(eps):
        try:
            D = np.load('YL_id='+str(id)+'_neps='+str(epochs)+'_L='+str(L)+'.npz')
            for k in keys:
                big_D[k+'s'][id,ne,:] = D[k]
        except Exception as e:
            print(e)
big_D = {k:np.squeeze(v) for k,v in big_D.items()}
big_D['epochs'] = eps
np.savez(savepath+'YL_sum_varySNMQ_L='+str(L),**big_D)
