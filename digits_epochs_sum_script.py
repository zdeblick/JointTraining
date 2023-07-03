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
trials = 40
shapes = [[d[0], in_shape[0]-2, in_shape[1]-2],[d[1], np.prod((in_shape-4)*5/4)],[d[2]]]
pen = 'cross_rc'
true_task_i=3
hypo_task_i=3
in_shape = np.array([8,6])    
N = int(d[0]*np.prod(in_shape-2)+d[1]*np.prod((in_shape-4))+d[1]*np.prod((in_shape-4)/2)+d[2])
epochs = [150,300,450,600,900,1200,1500,1800,2100,2400]

train_losses = np.nan*np.ones((trials,len(alphas),l1s.size,len(epochs)))
test_losses = np.nan*np.ones((trials,len(alphas),l1s.size,len(epochs),3))
cpfns = np.nan*np.ones((trials,len(alphas),l1s.size,len(epochs)))
alignments = np.nan*np.ones((trials,len(alphas),l1s.size,len(epochs)))
cms = np.nan*np.ones((trials,len(alphas),l1s.size,len(epochs),3,3))

trained_str = ['_untrained', '_eotrained', '_looptrained', ''][true_task_i]
matched_str = ['_L2matched', '_eomatched', '_loopmatched', ''][hypo_task_i]

for trial in range(trials):
    for e_i,n_eps in enumerate(epochs):
        for a_i, alpha in enumerate(alphas):
            for l_i, l1 in enumerate(l1s):
                try:
                    fname = 'digits_headlr_mednetmatch'+trained_str+matched_str+'_pen='+pen+'_trial'+str(trial)+'_ai='+str(a_i)+'_li='+str(l_i)+('_nepochs='+str(n_eps))*(n_eps!=300)
                    D = np.load(fname+'.npz',allow_pickle=True)
                    train_losses[trial,a_i,l_i,e_i] = D['train_loss']
                    test_losses[trial,a_i,l_i,e_i,:] = D['test_losses']
                    cpfns[trial,a_i,l_i,e_i] = D['cpfn']
                    alignments[trial,a_i,l_i,e_i] = D['alignment']
                    cms[trial,a_i,l_i,e_i,:,:] = D['layer_cm']
                except Exception as e:
                    print(e,trial,a_i,l_i,e_i)
                    stop
fname = savepath+'results_headlr_'+size+trained_str+matched_str+'_pen='+pen+'_epochs'
if not os.path.isfile(fname+'.npz') or True:
    np.savez(fname,train_losses=train_losses,test_losses=test_losses,
             cpfns=cpfns,alphas=alphas,l1s=l1s,alignments=alignments,d=d,cms=cms,epochs=epochs)
