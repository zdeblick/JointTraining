#!/home/daniel.zdeblick/anaconda3/bin/python3
import numpy as np
import os
from digits_sim_script import *

array_id_str = 'SLURM_ARRAY_TASK_ID' #slurm
#array_id_str = 'PBS_ARRAYID' #pbs

l1s = np.logspace(-6,-2,9) #lambda hyperparameter controlling C_map regularization
pen='cross_rc'
subsamples = [1]
epochs = [600, 900, 1800]
nonlinear = False

id = os.getenv(array_id_str)
id = 0 if id is None else int(id)
(trial,a_i,l_i,s_i,e_i) = np.unravel_index(id,(200,18,l1s.size,len(subsamples),len(epochs)))
true_task_i = 3
hypo_task_i = 3
sub = subsamples[s_i]
n_eps = epochs[e_i]
L2_matched = hypo_task_i==0
if L2_matched:
    alphas = [0.99999,0.9999,0.999,0.99,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.03,0.01,0.001,1e-4,1e-5]
else:
    alphas = [1-1e-5,1-1e-4,0.999,0.99,0.95,0.9,0.85,0.75,0.5,1e-3,1e-4,1e-5]
alpha = alphas[a_i]
l1 = l1s[l_i]
trained_str = ['_untrained', '_eotrained', '_looptrained', ''][true_task_i]
matched_str = ['_L2matched', '_eomatched', '_loopmatched', ''][hypo_task_i]
fname = 'digits_headlr_mednetmatch'+trained_str+matched_str+'_pen='+pen+'_trial'+str(trial)+'_ai='+str(a_i)+'_li='+str(l_i)+('_1000sub='+str(int(1000*sub)))*(sub<1)+('_nepochs='+str(n_eps))*(n_eps!=300)+'_linear'*(not nonlinear)
run_digits(fname,true_task_i,hypo_task_i,alpha,l1,sub,pen,trial,epochs=n_eps,nonlinear=nonlinear)
