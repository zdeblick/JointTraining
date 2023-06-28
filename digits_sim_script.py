import os
from sklearn import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import colorcet as cc
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression



class CustomDataset(Dataset):
    """A class that is used by DataLoader to supply input and output pairs to a pytorch network for training/evaluation"""

    def __init__(self, X, y):
        """
        Args:
            X: inputs
            y: outputs
        """
        self.X = np.array(X,dtype='float')
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if type(self.y) is list:
            return (self.X[idx],[self.y[i][idx] for i in range(len(self.y))])
        else:  
            return (self.X[idx],self.y[idx])


class Net(nn.Module):
    '''The convolutional neural network used as a data-generator (depicted in Figure 3)'''
    
    def __init__(self,in_shape = np.array([8,6]),d=[8,16,128,10]):
        '''
        in_shape: dimensions of input image
        d: dimensions of network (#channels in layer 1, #channels in layer 2, #hidden units in layer 3, #of classes in output)
        '''
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, d[0], 3, 1)
        self.conv2 = nn.Conv2d(d[0], d[1], 3, 1)
        self.fc1 = nn.Linear(int(d[1]*np.prod((in_shape-4)/2)), d[2])
        self.fc2 = nn.Linear(d[2], d[3])

    def forward(self, x):
        act = []
        x = self.conv1(x)
        x = F.relu(x)
        act.append(torch.flatten(x, 1))
        x = self.conv2(x)
        x = F.relu(x)
        mp = [torch.flatten(x, 2)]
        x = F.max_pool2d(x, 2)
        mp.append(torch.flatten(x, 2))
        act.append(torch.flatten(torch.cat(mp,dim=2),1))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        act.append(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
#         act.append(output)
        return output, torch.cat(act,1)


def train(dataloader, model, loss_fn, optimizer):
    '''
    Perform one epoch of training for a network
    Args:
        dataloader: DataLoader object that supples the training data
        model: Net object that is trained
        loss_fn: pytorch loss function applied to data for training (https://pytorch.org/docs/stable/nn.html#loss-functions)
        optimizer: pytorch Optimizer used for gradient descent (https://pytorch.org/docs/stable/optim.html)
    Returns:
        unit activites (u_i), a samples x units np.array
    '''
    size = len(dataloader.dataset)
    acts = []
    for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device) #specify device and uncomment to use GPU

        # Compute prediction error
        pred, act = model(X.float())
        acts.append(act.detach().numpy())
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return np.vstack(acts)
        
        
def test(dataloader, model, loss_fn):
    '''
    compute and print evaluation metrics on a model
    Args:
        dataloader: DataLoader object supplying evaluation data
        model: Net object that will be evaluated
        loss_fn: pytorch loss function applied to data for testing (https://pytorch.org/docs/stable/nn.html#loss-functions)
    Returns:
        unit activites (u_i), a samples x units np.array
    '''
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    acts = []
    with torch.no_grad():
        for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
            pred, act = model(X.float())
            acts.append(act.detach().numpy())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return np.vstack(acts)

class NetMatcher(nn.Module):
    '''
    Network model with the same structure as Net class, but which can be used for joint-training
    '''
    def __init__(self,in_shape = np.array([8,6]), d=[8,16,128,10],N_meas=None):
        super(NetMatcher, self).__init__()
        self.conv1 = nn.Conv2d(1, d[0], 3, 1)
        self.conv2 = nn.Conv2d(d[0], d[1], 3, 1)
        self.fc1 = nn.Linear(int(d[1]*np.prod((in_shape-4)/2)), d[2])
        self.fc2 = nn.Linear(d[2], d[3])
        N = int(d[0]*np.prod(in_shape-2)+d[1]*np.prod((in_shape-4))+d[1]*np.prod((in_shape-4)/2)+d[2])
        if N_meas is None:
            N_meas = N
        self.fc_act = nn.Linear(N,N_meas)
#         self.fc_act.weight = torch.nn.Parameter(self.fc_act.weight.data.to_sparse())
#         torch.nn.init.uniform_(self.fc_act.weight)

    def forward(self, x):
        act = []
        x = self.conv1(x)
        x = F.relu(x)
        act.append(torch.flatten(x, 1))
        x = self.conv2(x)
        x = F.relu(x)
        mp = [torch.flatten(x, 2)]
        x = F.max_pool2d(x, 2)
        mp.append(torch.flatten(x, 2))
        act.append(torch.flatten(torch.cat(mp,dim=2),1))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        act.append(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
#         act.append(output)
        act_output = self.fc_act(torch.cat(act,1))
        return output, act_output
    

def joint_loss(pred,y,lossfns,alpha=1.0):
    '''
    cost function used for joint-training
    Args:
        pred: length-2 list of predicted model ouputs for hypothesized task and neural responses
        y: length-2 list of target ouputs for hypothesized task and neural responses
        lossfns: length-2 list of pytorch loss functions for hypothesized task and neural responses
        alpha: hyperparameter weighting these two cost functions (alpha = beta/(1+beta))
    Returns:
        joint cost function (pytorch loss function)
    '''
    return alpha*lossfns[0](pred[0],y[0])+(1-alpha)*lossfns[1](pred[1],y[1])

def L2reg_loss(pred,y,loss,model,alpha=1.0):
    '''
    cost function using L2 regularization instead of a hypothesized task
    Args:
        pred: predicted model ouputs for neural responses
        y: target ouputs for neural responses
        loss: pytorch loss function for neural responses
        model: NetMatcher object with joint-trained parameters
        alpha: hyperparameter weighting these two cost functions (alpha = beta/(1+beta))
    Returns:
        cost function (pytorch loss function)
    '''
    L2_reg = 0
    count = 0
    for p in set(model.parameters())-set(model.fc2.parameters())-set(model.fc_act.parameters()):
        L2_reg += torch.norm(p,2)
        count+=1
    return alpha*L2_reg/count+(1-alpha)*loss(pred,y)


def make_sparse_hard(model2):
    '''applies hard threshholding to a NetMatcher object model2 as described in eq S12'''
    W = model2.fc_act.weight.data
    vals,inds = torch.max(W,axis=1,keepdims=False)
    rep = torch.zeros(W.shape)
    rep[np.arange(W.shape[0]),inds] = vals
    model2.fc_act.weight.data = rep
    model2.fc_act.weight.grad[rep==0] = 0

def make_sparse(model2):
    '''an alternative threshholding scheme not used in the paper'''
    W = model2.fc_act.weight.data
    thresh=1e-2
    model2.fc_act.weight.data = torch.where(W>thresh,W,torch.zeros((1,)))
    model2.fc_act.weight.grad[W<thresh] = 0


def train_joint(dataloader, model, lossfns, optimizer, alpha=1.0, l1=0.0, freeze_small=False, pen = 'cross_rc',L2_matched=True):
    '''
    perform one epoch of joint-training
    Args:
        dataloader: a DataLoader object that supplies the training data
        model: an untrained Net object representing the model to be used for joint-training - modified by the function during training
        lossfns: length-2 list of pytorch loss functions for hypothesized task and neural responses
        optimizer: pytorch Optimizer used for gradient descent (https://pytorch.org/docs/stable/optim.html)
        alpha: (float) hyperparameter weighting these two cost functions (alpha = beta/(1+beta))
        l1: (float) hyperparameter weighting C_map
        freeze_small: boolean determining whether or not thresholding is applied to theta_y
        pen: (str) what function to use for C_map (only 'cross_rc' used in paper)
        L2_matched: boolean determining whether to use L2 regularization instead of hypothesized computation
    Returns:
        unit activites (u_i), a samples x units np.array
        epoch_loss, value of the joint cost function
    '''
    size = len(dataloader.dataset)
    acts = []
    epoch_loss = 0
    for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred, act = model(X.float())
        acts.append(act.detach().numpy())
        if L2_matched:
            loss = L2reg_loss(act, y[1], lossfns[1], model, alpha=alpha)
        else:
            loss = joint_loss((pred,act), y, lossfns,alpha=alpha)
        W = model.fc_act.weight
        if pen=='cross_rc':
            W = torch.abs(W)
            penalty = torch.norm(torch.mul(W,torch.sum(W,1,keepdims=True)+torch.sum(W,0,keepdims=True)-2*W),1)
        elif pen=='cross_r':
            penalty = torch.norm(torch.mul(W,torch.sum(W,1,keepdims=True)-W),1)
        elif pen=='l1':
            penalty = torch.norm(model.fc_act.weight,1)
        elif pen is None:
            penalty = 0
        loss = loss (1-alpha)*l1*penalty


        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        if freeze_small:
            #make_sparse(model)
            make_sparse_hard(model)
        optimizer.step()
        
        #nonegativity constraint
        W = model.fc_act.weight.data
        if batch == 0:
            W = W.abs()
        model.fc_act.weight.data = W.clamp(min=0.0)
        
        batch_loss, current = loss.item(), batch * len(X)
        epoch_loss+=batch_loss/len(X)
        if batch == size//len(X)-1:
            print(f"loss: {epoch_loss:>7f}")
#             print(f"loss: {batch_loss:>7f}  [{current:>5d}/{size:>5d}]")
    return np.vstack(acts), epoch_loss
        
        
def test_joint(dataloader, model, lossfns, L2_matched):
    '''
    evaluate a joint-trained model
    Args:
        dataloader: a DataLoader object that supplies the evaluation data
        model: a Net object representing the joint-trained model
        lossfns: length-2 list of pytorch loss functions for hypothesized task and neural responses
        L2_matched: boolean determining whether to use L2 regularization instead of hypothesized computation
    Returns:
        unit activites (u_i), a samples x units np.array
        tuple of (test_loss1,test_loss2,correct), the losses for the task and responses followed by the class identification accuracy
    '''
    size = len(dataloader.dataset)
    model.eval()
    test_loss1, test_loss2, correct = 0, 0, 0
    acts = []
    with torch.no_grad():
        for X, y in dataloader:
            #X, y = X.to(device), y.to(device)
            pred, act = model(X.float())
            acts.append(act.detach().numpy())
            if not L2_matched:
                test_loss1 += lossfns[0](pred, y[0]).item()
            test_loss2 += lossfns[1](act, y[1]).item()
            correct += (pred.argmax(1) == y[0]).type(torch.float).sum().item()
        if L2_matched:
            L2_reg = 0
            cnt = 0
            for p in set(model.parameters())-set(model.fc2.parameters())-set(model.fc_act.parameters()):
                L2_reg += torch.norm(p,2)
                cnt+=1
            test_loss1 = L2_reg/cnt
        else:
            test_loss1 /= size
    test_loss2 /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Loss1: {test_loss1:>8f}, Loss2: {test_loss2:>8f} \n")
    return np.vstack(acts), (test_loss1,test_loss2,correct)


def layer_cm(W,layer_sizes):
    '''
    Computes the normalized layer-wise confusion matrix (W_ij in the paper).  
    Args:
        W: fitted map between model units and artificial neurons (theta_y in the paper)
        layer_sizes: list of integers that specify the size, in units, of each layer of the model/data-generator. 
                    It is assumed that the rows and columns of W (theta_y) are organized in blocks that correspond to these layers
    Returns:
        cm: layer-wise confusion matrix. Note that a different normalization was used here than that in the paper - this is corrected in Figures.ipynb
    '''
    cm = np.zeros((len(layer_sizes),len(layer_sizes)))
    r = 0
    for r_l, r_l_size in enumerate(layer_sizes):
        c = 0
        for c_l, c_l_size in enumerate(layer_sizes):
            cm[r_l,c_l] = np.sum(W[r:r+r_l_size,c:c+c_l_size])/np.sqrt(r_l_size*c_l_size)
            c += c_l_size
        r += r_l_size
    return cm


def alignment(W,shapes):
    '''
    a function that attempts to account for invariances in the model architecture to compute a finer resolution accuracy metric. Not used in the paper
    Args:
        W: fitted map between model units and artificial neurons (theta_y in the paper)
        layer_sizes: list of integers that specify the size, in units, of each layer of the model/data-generator. 
                    It is assumed that the rows and columns of W (theta_y) are organized in blocks that correspond to these layers
    Returns:
        alignment metric
    '''
    s0 = 0
    counted = 0
    score = 0
    c_all = []
    for shape in shapes:
        s1 = np.prod(shape)
        block_size = s1//shape[0]
        W_l = W[s0:s0+s1,s0:s0+s1]
        if block_size==1:
            block_scores = W_l
        else:
            block_scores = np.zeros((shape[0],shape[0]))
            for r in range(shape[0]):
                for c in range(shape[0]):
                    block = W_l[block_size*r:block_size*(r+1),block_size*c:block_size*(c+1)]
                    block_scores[r,c] = np.sum(np.diag(block))
        r_inds,c_inds = linear_sum_assignment(-1.0*block_scores)
        c_all.append(c_inds)
        score += 2*np.sum(block_scores[r_inds,c_inds])-np.sum(np.ravel(W_l))
        counted += np.sum(np.ravel(W_l))
        s0+=s1
    total = np.sum(np.ravel(W))
    print(score,counted,total)
    return (score - (total - counted))/total



def run_digits(fname,true_task_i,hypo_task_i,alpha,l1,sub,pen,trial):
    os.chdir('digits_results')

    #Load the digits dataset
    digits = datasets.load_digits()
    images = np.expand_dims(digits.images[:,:,1:-1],1)
    in_shape = np.array([8,6],dtype='int')
    n_train = 1024 # number of samples in training set
    d = [16,32,128,10]
    model = Net(in_shape=in_shape, d=d)
    model = model.float()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    L2_matched = hypo_task_i==0

    np.random.seed(1234567)
    torch.manual_seed(1234567)

    shapes = [[d[0], in_shape[0]-2, in_shape[1]-2],[d[1], np.prod((in_shape-4))],[d[2]]]
    print(shapes)
    shapes = [[d[0], 6, 4],[d[1], 10],[d[2]]]
    print(shapes)

    task_maps = [(lambda x: x), (lambda x: x%2), (lambda x: np.isin(x,[0,6,8,9]).astype(int)), (lambda x: x)]

    # Create data loaders for training the data-generator.
    training_data = CustomDataset(images[:n_train,:,:,:],task_maps[true_task_i](digits.target[:n_train]))
    test_data = CustomDataset(images[n_train:,:,:,:],task_maps[true_task_i](digits.target[n_train:]))
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)


    if true_task_i>0:
        #train data-generator
        epochs = 60
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_acts = train(train_dataloader, model, loss_fn, optimizer)
            test_acts = test(test_dataloader, model, loss_fn)
    else:
        #use untrained network
        train_acts = test(train_dataloader, model, loss_fn)
        test_acts = test(test_dataloader, model, loss_fn)

    #compute predictability on validation data of each possible task z_i from simulated responses y_i
    Cs = np.logspace(-3,4,22)
    # metrics x possible z_i's x l2 regularization
    task_alignments = np.zeros((2,len(task_maps)-1,Cs.size)) 
    for task_i in range(1,len(task_maps)):
        for Ci,C in enumerate(Cs):
            LR = LogisticRegression(C=C)
            LR.fit(train_acts,task_maps[task_i](digits.target[:n_train]))
            #negative log-likelihood
            task_alignments[0,task_i-1,Ci] = LR.score(test_acts,task_maps[task_i](digits.target[n_train:])) 
            #accuracy
            task_alignments[1,task_i-1,Ci] = np.mean(LR.predict_log_proba(test_acts)[np.arange(test_acts.shape[0]),task_maps[task_i](digits.target[n_train:]).astype(int)])


    print("Done training!")
    np.random.seed(trial)
    torch.manual_seed(trial)

    #randomly choose a fraction of simulated neurons for use as data
    N = train_acts.shape[1]
    if sub is not None and sub<1:
        fake_neurons_measured = np.random.choice(N,size=int(N*sub),replace=False)
        train_acts = train_acts[:,fake_neurons_measured]
        test_acts = test_acts[:,fake_neurons_measured]
    else:
        fake_neurons_measured = np.arange(N)
    N_meas = fake_neurons_measured.size

    # Create training and validation data loaders for joint-training
    training_data_both = CustomDataset(images[:n_train,:,:,:],[task_maps[hypo_task_i](digits.target[:n_train]),train_acts])
    test_data_both = CustomDataset(images[n_train:,:,:,:],[task_maps[hypo_task_i](digits.target[n_train:]),test_acts])

    train_dataloader2 = DataLoader(training_data_both, batch_size=batch_size)
    test_dataloader2 = DataLoader(test_data_both, batch_size=batch_size)

    # initialize the joint-training model and partition its parameters into those for theta_u (base_params), theta_y (act_head_params), and theta_z (task_head_params)
    model2 = NetMatcher(d=d,N_meas=N_meas)
    model2 = model2.float()
    all_params = set(model2.parameters())
    task_head_params = set(model2.fc2.parameters())
    act_head_params = set(model2.fc_act.parameters())
    base_params = all_params-task_head_params-act_head_params
    base_params = list(base_params)
    task_head_params = list(task_head_params)
    act_head_params = list(act_head_params)
    # initialize the optimizer and learning schedule, with modified learning rates for the different thetas as described in Section S2
    init_lr=1e-3
    optimizer2 = torch.optim.Adam([{'params':base_params},{'params':task_head_params,'lr':init_lr/alpha},{'params':act_head_params,'lr':init_lr/(1-alpha)}], lr=init_lr)
    epochs = 300
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2,verbose=True,factor=0.5,patience=20,min_lr = 1e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2,epochs,verbose=False,eta_min = 1e-4)

    lossfns = (F.cross_entropy, F.mse_loss)

    #Perform joint-training
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        if t < 150 or pen is None:
            # for the first 150 epochs, apply C_map regularization, no thresholding, keep learning rates for theta_y and theta_z fixed
            _, loss = train_joint(train_dataloader2, model2, lossfns, optimizer2, alpha=alpha, l1=l1, pen=pen, L2_matched=L2_matched)
            loss+=30 #prevents scheduler from being tripped at epoch 150
            scheduler.step(loss)
            optimizer2.param_groups[1]['lr'] = init_lr/alpha
            optimizer2.param_groups[2]['lr'] = init_lr/(1-alpha)
        else:
            # for the last 150 epochs, no C_map regularization, instead threshholding
            _, loss = train_joint(train_dataloader2, model2, lossfns, optimizer2, alpha=alpha,freeze_small=True,L2_matched=L2_matched)
            scheduler.step(loss)
        _, losses = test_joint(test_dataloader2, model2, lossfns, L2_matched)
    # add back in rows that were not ''recorded'' for consistency
    W_part = model2.fc_act.weight.detach().numpy()
    W = np.zeros((N,N))
    W[fake_neurons_measured,:] = W_part
    algn = alignment(np.abs(W)>0,shapes) if pen is not None else None
    lcm = layer_cm(np.abs(W)>0,[np.prod(s) for s in shapes]) if pen is not None else layer_cm(np.abs(W),[np.prod(s) for s in shapes])
    np.savez(fname,train_loss=loss,test_losses=losses,task_alignments=task_alignments,
        sparsity=-12,cpfn=np.sum(np.ravel(W)>0)/train_acts.shape[1],alignment=algn,layer_cm=lcm)

    print('Done',fname)
