import numpy as np
import os

os.chdir('YL_results')
savepath = '../YL_summaries/'

L=4
inds = 20000
epochs = 200
D0 = np.load('YL_LBFGS_id=0_neps='+str(epochs)+'_L='+str(L)+'.npz')
keys = D0.keys()
big_D = {k+'s':np.nan*np.ones(np.hstack((inds,D0[k].shape)).astype(int) ) for k in keys}
for id in range(inds):
    try:
        D = np.load('YL_LBFGS_id='+str(id)+'_neps='+str(epochs)+'_L='+str(L)+'.npz')
        for k in keys:
            big_D[k+'s'][id] = D[k]
    except Exception as e:
        print(e)

eps = (0,1,4,9,19,49,99,199)
big_D = {k:np.squeeze(v) for k,v in big_D.items()}
big_D['Cvalss'] = big_D['Cvalss'][:,:,eps]
big_D['eps'] = eps
np.savez(savepath+'YL_sum_varySNMQ_L='+str(L),**big_D)
