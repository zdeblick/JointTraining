import numpy as np
import os

os.chdir('YL_results')
savepath = '../YL_summaries/'

L=4
inds = 325
epochs = 20000
D0 = np.load('YL_id=0_neps='+str(epochs)+'_L='+str(L)+'.npz')
keys = D0.keys()
big_D = {k+'s':np.nan*np.ones(np.hstack((inds,np.expand_dims(D0[k],0).shape))) for k in keys}
for id in range(inds):
    try:
        D = np.load('YL_id='+str(id)+'_neps='+str(epochs)+'_L='+str(L)+'.npz')
        for k in keys:
            big_D[k+'s'][id,:] = D[k]
    except Exception as e:
        print(e)
big_D = {k:np.squeeze(v) for k,v in big_D.items()}
big_D['epochs'] = np.arange(100,epochs+1,100)
np.savez(savepath+'YL_sum_varySNMQ_L='+str(L),**big_D)
