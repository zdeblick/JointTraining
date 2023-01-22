import os
import numpy as np

os.chdir('digits_results')
savepath = '../digits_summaries/'

in_shape = np.array([8,6],dtype='int')
d = [16,32,128,10]
size='med'

#size='small'
#d = [8,16,128,10]

alphas = [1-1e-5,1-1e-4,0.999,0.99,0.95,0.9,0.85,0.75,0.5,1e-3,1e-4,1e-5]
#alphas = [1-1e-6,1-1e-5,1-1e-4,0.999,0.995,0.99,0.98,0.97,0.95,1e-5]
l1s = np.logspace(-6,-2,9)
subsets = [0.05,0.1,0.25,0.5,1]
trials = 40
shapes = [[d[0], in_shape[0]-2, in_shape[1]-2],[d[1], np.prod((in_shape-4)*5/4)],[d[2]]]
pen = 'cross_rc'
eo_trained = False
eo_matched = False
in_shape = np.array([8,6])    
N = int(d[0]*np.prod(in_shape-2)+d[1]*np.prod((in_shape-4))+d[1]*np.prod((in_shape-4)/2)+d[2])


train_losses = np.nan*np.ones((trials,len(alphas),l1s.size,len(subsets)))
test_losses = np.nan*np.ones((trials,len(alphas),l1s.size,len(subsets),3))
cpfns = np.nan*np.ones((trials,len(alphas),l1s.size,len(subsets)))
alignments = np.nan*np.ones((trials,len(alphas),l1s.size,len(subsets)))
cms = np.nan*np.ones((trials,len(alphas),l1s.size,len(subsets),3,3))

for trial in range(trials):
    for s_i,sub in enumerate(subsets):
        subtext = '_1000sub='+str(int(1000*sub)) if sub<1 else ''
        for a_i, alpha in enumerate(alphas):
            for l_i, l1 in enumerate(l1s):
                try:
                    D = np.load('files/digits_headlr_'+size+'netmatch'+'_eotrained'*eo_trained+'_eomatched'*eo_matched+'_pen='+pen+'_trial'+str(trial)+'_ai='+str(a_i)+'_li='+str(l_i)+subtext+'.npz',allow_pickle=True)
                    train_losses[trial,a_i,l_i,s_i] = D['train_loss']
                    test_losses[trial,a_i,l_i,s_i,:] = D['test_losses']
                    cpfns[trial,a_i,l_i,s_i] = D['cpfn'] if sub<1 else (1-D['sparsity'])*N
                    alignments[trial,a_i,l_i,s_i] = D['alignment']
                    cms[trial,a_i,l_i,s_i,:,:] = D['layer_cm']
                except Exception as e:
                    print(e,trial,a_i,l_i,s_i)
                    stop
fname = savepath+'results_headlr_'+size+'_eotrained'*eo_trained+'_eomatched'*eo_matched+'_pen='+pen+'_thresh_hard_subsets'
if not os.path.isfile(fname+'.npz') or True:
    np.savez(fname,train_losses=train_losses,test_losses=test_losses,
             cpfns=cpfns,alphas=alphas,l1s=l1s,alignments=alignments,d=d,cms=cms,subsets=subsets)
