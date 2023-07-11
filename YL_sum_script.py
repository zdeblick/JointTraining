import numpy as np
import os

os.chdir('YL_results')
savepath = '../YL_summaries/'

inds = 20000
eps = [1000,5000]
keys = np.load('YL_id=0_neps=5000.npz').keys()
big_D = {k+'s':np.zeros((inds,len(eps))) for k in keys}
for id in range(inds):
    for ne, epochs in enumerate(eps):
        D = np.load('YL_id='+str(id)+'_neps='+str(epochs)+'.npz')
        for k in keys:
            big_D[k+'s'][id,ne] = D[k]

np.savez(savepath+'YL_sum_varySNMQ',**big_D)
