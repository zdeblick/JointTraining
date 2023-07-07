import numpy as np
import os

os.chdir('YL_results')
savepath = '../YL_summaries/'

inds = 20000
keys = np.load('YL_id=0.npz').keys()
big_D = {k+'s':np.zeros((20000,)) for k in keys}
for id in range(inds):
    D = np.load('YL_id='+str(id)+'.npz')
    for k in keys:
        big_D[k+'s'][id] = D[k]

np.savez(savepath+'YL_sum_varySNMQ',**big_D)
