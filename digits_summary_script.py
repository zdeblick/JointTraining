import os
import numpy as np
import itertools

os.chdir('digits_results')
savepath = '../digits_summaries/'

in_shape = np.array([8,6],dtype='int')
d = [16,32,128,10]
size='med'
pen = 'l1'
#pen = 'cross_rc'
#pen = None
thresh = 'soft'
threshstr = '' if thresh is None else '_thresh='+thresh

#size='small'
#d = [8,16,128,10]

l1s = np.array([0]) if pen is None else np.logspace(-6,-2,9)
trials = 40
shapes = [[d[0], in_shape[0]-2, in_shape[1]-2],[d[1], np.prod((in_shape-4)*5/4)],[d[2]]]


#for true_task_i,hypo_task_i in ((3,3),): 
for true_task_i,hypo_task_i in itertools.product(range(4),range(4)):
    trained_str = ['_untrained', '_eotrained', '_looptrained', ''][true_task_i]
    matched_str = ['_L2matched', '_eomatched', '_loopmatched', ''][hypo_task_i]
    L2_matched = hypo_task_i==0

    if L2_matched:
        alphas = [0.99999,0.9999,0.999,0.99,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.03,0.01,0.001,1e-4,1e-5]
    else:
        alphas = [1-1e-5,1-1e-4,0.999,0.99,0.95,0.9,0.85,0.75,0.5,1e-3,1e-4,1e-5]

    train_losses = np.nan*np.ones((trials,len(alphas),l1s.size))
    test_losses = np.nan*np.ones((trials,len(alphas),l1s.size,3))
    sparsities = np.nan*np.ones((trials,len(alphas),l1s.size))
    alignments = np.nan*np.ones((trials,len(alphas),l1s.size))
    cms = np.nan*np.ones((trials,len(alphas),l1s.size,3,3))
    task_alignments = np.nan*np.ones((trials,len(alphas),l1s.size,2,3,22))

    for trial in range(trials):
        for a_i, alpha in enumerate(alphas):
            for l_i, l1 in enumerate(l1s):
                try:
                    fname = 'digits_headlr_'+size+'netmatch'+trained_str+matched_str+('' if pen is  None else '_pen='+pen)+threshstr+'_trial'+str(trial)+'_ai='+str(a_i)+'_li='+str(l_i)
                    D = np.load(fname+'.npz',allow_pickle=True)
                    train_losses[trial,a_i,l_i] = D['train_losses'][1]
                    test_losses[trial,a_i,l_i,:] = D['test_losses']
                    sparsities[trial,a_i,l_i] = D['cpfn']
                    if D['alignment']!=None:
                        alignments[trial,a_i,l_i] = D['alignment']
                    cms[trial,a_i,l_i,:,:] = D['layer_cm']
                    task_alignments[trial,a_i,l_i,:,:,:] = D['task_alignments']
                except Exception as e:
                    print(e,trial,a_i,l_i)
    fname = savepath+'results_headlr_'+size+trained_str+matched_str+('' if pen is  None else '_pen='+pen)+threshstr
    if not os.path.isfile(fname+'.npz') or True:
        np.savez(fname,train_losses=train_losses,test_losses=test_losses,task_alignments=task_alignments,
                 sparsities=sparsities,alphas=alphas,l1s=l1s,alignments=alignments,d=d,cms=cms)
