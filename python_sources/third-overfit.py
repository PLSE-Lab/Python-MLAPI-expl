#!/usr/bin/env python
# coding: utf-8

# Just for fun:)
# 
# Using knowledge gained by competing in quora incencere questions:
# 
# https://www.kaggle.com/xsakix/pytorch-bilstm-meta-v2
# 
# # What to do?
# 
# What if I would to change the activation function?
# 
# What if I would to change the optimizer?
# 
# What if I would to change the learning rate?
# 
# What if I would to change the weight decay?
# 
# What if I would to change the momentum?
# 
# **What if I could do this whole thing in some kind of grid search???
# **
# 
# What if I would to use a cnn to filter "important" features?
# 
# What if I would to use gru/lstm to learn if features have sequential dependencies?
# 
# What if I would to use some kind of bayesian feature selector?
# 
# What if I would to use some kind of weighted feature selection?
# 
# Humans can generalize from limited number of samples. What would happen if I would to use RL for this problem? Could a RL alg learn to generalize this problem?
# 
# Remarks:
# 
# 2019-02-19:
# Looks like a good starting point for improving score.
# Higher validation aucroc leads to higher LB.
# 
# 2019-02-20:
# 
# Everything is nondeterministic. I don't know currently how to make this kernel deterministic. I tried to turn off GPU and cuda, but it doesn't work. 
# 
# Could It be that the amount of data for training is so low, that it can't be deterministic? (i don't believe this)
# Could It be that pytorch is nodeterministic? (also i don't believe this)
# What's the reason? How can I make this deterministic?
# 
# Solution:
# seeding needs to be done before each model training. Simple test is:
# 
# seed_everything(seed)
# print(np.random.rand(10))
# seed_everything(seed)
# print(np.random.rand(10))
# 
# vs
# 
# print(np.random.rand(10))
# print(np.random.rand(10))
# 
# 21-02-2019:
# 
# Did a CV over activation functions, optimizers, learning rates and weight decay, but the resulting "setup" gained only 0.561 on LB.
# This means that a good validation AUCROC doesn't lead in all cases to a good LB.
# 
# Would a situation in which training AUCROC and validation AUCROC differ only little be better in this case? If they differ too much (what is too much?) does it mean it can't generalize properly?
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import random
import torch

seed = 12

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything(seed)
print('Seeding done...')


# In[ ]:


df_train = pd.read_csv("../input/train.csv")
train_y = df_train.target.values
train_x = df_train.drop(columns=['id','target']).values
print(train_x[:5])
print('-'*80)
print(train_y[:5])
print('-'*80)
print(train_y.shape)
print(train_x.shape)
print('-'*80)

df_test = pd.read_csv("../input/test.csv")
print(df_test.shape)
test_x = df_test.drop(columns=['id']).values
print(test_x.shape)
print('-'*80)


# In[ ]:


# oversampling doesnt work
# from imblearn.over_sampling import RandomOverSampler,ADASYN, SMOTE, SMOTENC

# resampled_train_x1, resampled_train_y1 = RandomOverSampler(random_state=seed).fit_resample(train_x, train_y)
# resampled_train_x2, resampled_train_y2 = ADASYN(random_state=seed).fit_resample(train_x, train_y)
# resampled_train_x3, resampled_train_y3 = SMOTE(random_state=seed).fit_resample(train_x, train_y)
# resampled_train_x4, resampled_train_y4 = SMOTENC(random_state=seed,categorical_features=[0,1]).fit_resample(train_x, train_y)
# resampled_train_x = np.concatenate([resampled_train_x1,resampled_train_x2,resampled_train_x3,resampled_train_x4])
# resampled_train_y = np.concatenate([resampled_train_y1,resampled_train_y2,resampled_train_y3,resampled_train_y4])
# print(resampled_train_x.shape)
# print(resampled_train_y.shape)


# In[ ]:


# takes too long, not feasible
# import pymc3 as pm

# betas = []

# with pm.Model() as model:
#     sigma = pm.Uniform(name='sigma', lower=np.min(train_x), upper=np.max(train_y))
#     for i in range(train_x.shape[1]):
#         betas.append(pm.Normal(name='b'+str(i), mu=mu_x[i], sd=std_x[i]))
#     mu = pm.Deterministic('mu', sum([betas[i]*train_x[:,i] for i in range(train_x.shape[1])]))
#     target = pm.Normal(name='target', mu=mu, sd=sigma, observed=train_y)
#     trace_model = pm.sample(1000, tune=1000)


    


# In[ ]:


#src: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss dosen't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


# In[ ]:


from sklearn.metrics import roc_auc_score
#https://www.kaggle.com/shujian/single-rnn-with-4-folds-v1-9
def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(100)]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = roc_auc_score(y_true=y_true, y_score=y_proba > threshold)
#         print('\rthreshold = %f | score = %f'%(threshold,score),end='')
        if score > best_score:
            best_threshold = threshold
            best_score = score
#     print('\nbest threshold is % f with score %f'%(best_threshold,best_score))
    search_result = {'threshold': best_threshold, 'AUCROC': best_score}
    return search_result


# In[ ]:


# generates same data
# X_train1,X_val,y_train1, y_val = train_test_split(train_x,train_y,random_state=seed,stratify=train_y)
# X_train2,X_val2,y_train2, y_val2 = train_test_split(train_x,train_y,random_state=seed,stratify=train_y)

# print(X_train1[X_train1 != X_train2])
# print(X_val[X_val != X_val2])
# print(y_train1[y_train1 != y_train2])
# print(y_val[y_val != y_val2])


# In[ ]:


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchtext.data
import warnings
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import warnings
from sklearn.model_selection import StratifiedKFold,train_test_split

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

class OverfitModel(nn.Module):
    
    def __init__(self,activation,size):
        super(OverfitModel,self).__init__()
        self.classifier = nn.Sequential(            
            nn.Linear(300,size, bias=False),
            activation,
            nn.Dropout(0.5),
            nn.Linear(size,1,bias=False)
        );

    def forward(self,x):
        return self.classifier(x)


def eval_on_set(model,test_loader,loss_function):
    pred = []
    avg_loss = 0.
    model.eval()
    with torch.no_grad():
        for batch,(x_test_batch,y_test_batch) in enumerate(list(test_loader),1):
            y_pred = model(x_test_batch).squeeze(1)
            pred += torch.sigmoid(y_pred).cpu().detach().numpy().tolist()
            loss = loss_function(y_pred,y_test_batch)
            avg_loss += loss.item()
            
    return np.array(pred),avg_loss/batch


def train(model, train_loader,optimizer,loss_function ):    
    
    model.train()
    avg_loss = 0
    pred = []
    for batch,(x_batch,y_true) in enumerate(list(iter(train_loader)),1):
        optimizer.zero_grad()

        y_pred = model(x_batch).squeeze(1)
        pred += torch.sigmoid(y_pred).cpu().detach().numpy().tolist()
        loss = loss_function(y_pred,y_true)
        avg_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    return np.array(pred),avg_loss/batch

def eval_sub(model,submission_loader):

    pred = []
    model.eval()
    with torch.no_grad():
        for (x,) in list(submission_loader):       
            y_pred = torch.sigmoid(model(x).squeeze(1)).detach()
            pred += y_pred.cpu().numpy().tolist()

    return np.array(pred)

class Stats:
    def __init__(self):
        self.split_losses = []
        self.split_aucrocs = []
        self.split_val_losses = []
        self.split_val_aucrocs = []
        
        self.model_split_losses = []
        self.model_split_aucrocs = []
        self.model_split_val_losses = []
        self.model_split_val_aucrocs = []
        
    def clear_model_stat(self):
        self.model_split_losses = []
        self.model_split_aucrocs = []
        self.model_split_val_losses = []
        self.model_split_val_aucrocs = []
            
    def append_train_epoch(self,loss,auroc):
        self.model_split_losses.append(loss)
        self.model_split_aucrocs.append(auroc)
    
    def append_val_epoch(self,loss,auroc):
        self.model_split_val_losses.append(loss)
        self.model_split_val_aucrocs.append(auroc)
    
    def append_model(self):
        self.split_losses.append(self.model_split_losses)
        self.split_aucrocs.append(self.model_split_aucrocs)
        self.split_val_losses.append(self.model_split_val_losses)
        self.split_val_aucrocs.append(self.model_split_val_aucrocs)

def init_fc(worker_id):
    np.random.seed(seed)

def create_data_loader(X,Y,batch_size=50):
#     x_tensor = torch.tensor(X, dtype=torch.float32).cuda()
#     y_tensor = torch.tensor(Y, dtype=torch.float32).cuda()
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(Y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x_tensor,y_tensor)
    return torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True, worker_init_fn=init_fc, num_workers=0)
    
def create_submission_loader(X,batch_size=50):
#     submission_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32).cuda())
    submission_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32))
    submission_loader = torch.utils.data.DataLoader(dataset=submission_dataset,batch_size=batch_size, shuffle=False)
    return submission_loader

def one_cv(model,optimizer):
    seed_everything(seed)
#     model = model.cuda()
    model.apply(weights_init)
    batch_size=50
    
    stats = Stats()
#     loss_function = nn.BCEWithLogitsLoss().cuda()        
    loss_function = nn.BCEWithLogitsLoss()
    early_stop = EarlyStopping(patience=2)

    X_train1,X_val,y_train1, y_val = train_test_split(train_x,train_y,random_state=seed,stratify=train_y)
    train_loader =create_data_loader(X_train1,y_train1,batch_size)
    val_loader =create_data_loader(X_val,y_val,batch_size)

    stats.clear_model_stat()
    for epoch in range(20):
        y_pred,loss = train(model, train_loader,optimizer,loss_function)
        search = threshold_search(y_train1,y_pred)
        aucroc = search['AUCROC']
        stats.append_train_epoch(loss,aucroc)

        y_pred, val_loss = eval_on_set(model,val_loader, loss_function)
        search = threshold_search(y_val,y_pred)
        val_aucroc = search['AUCROC']
        stats.append_val_epoch(val_loss,val_aucroc)

#         print('EPOCH: ',epoch,': loss :',loss,': aucroc : ',aucroc,' : val loss: ',val_loss, ': val aucroc: ',val_aucroc)
#         print('-'*80)

        early_stop(np.round(1.-search['AUCROC'],decimals=5),model)
        if early_stop.early_stop:
            break

    stats.append_model()

#     print('FINISHED TRAINING META...')
    #load best performing
    model.load_state_dict(torch.load('checkpoint.pt'))

    y_pred,_ = eval_on_set(model,train_loader, loss_function)
    search = threshold_search(y_train1,y_pred)
    train_aucroc = search['AUCROC']
#     print('AUCROC:',train_aucroc)

    y_pred,_ = eval_on_set(model,val_loader, loss_function)
    search = threshold_search(y_val,y_pred)
    val_aucroc = search['AUCROC']
#     print('VAL AUCROC:',val_aucroc)
    score = np.abs(val_aucroc-train_aucroc)/(val_aucroc*100.)
    if val_aucroc < 0.7:
        # add penalty for any less then 0.6 
        score+=1.
#     print('SCORE:', score)
    
    
    return score,stats,search


# In[ ]:


activations = [
#     OverfitModel(nn.AdaptiveLogSoftmaxWithLoss(in_features=300,n_classes=2,cutoffs= [1])),
    'nn.ReLU',
    'nn.Tanh',
    'nn.Sigmoid',    
    'nn.Softmax',    
]

requeres_dim = [
    'nn.Softmin',    
    'nn.Softmax',    
    'nn.LogSoftmax']

# for threshold in [i * 0.01 for i in range(100)]:
#     models.append(OverfitModel(nn.Hardshrink(threshold)))

optimizers =[
    'optim.Adam',
    'optim.RMSprop',
    'optim.Adadelta',
    'optim.Adagrad',
    'optim.Adamax',
    'optim.ASGD'
]

lrs = [1,1e-1,1e-2,1e-3,1e-4]
decays = [1,1e-1,1e-2,1e-3,1e-4,1e-5]
networks = [16,32,64,128,256,512,1024]

results = {}

#inefficient...
for a in activations:
    for o in optimizers:
        for lr in lrs:
            for decay in decays:
                for size in networks:
                    if a not in requeres_dim:
                        activation = eval(a)()
                    else:
                        activation = eval(a)(dim=1)
                    model = OverfitModel(activation,size)
#                     print(a,':',o,':',lr,':',decay)
                    optimizer = eval(o)(model.parameters(),lr=lr,weight_decay=decay)
                    score,_,_ = one_cv(model,optimizer)
                    if score >= 1.:
                        continue
                    results[score] = {'size':size,'activation':a,'optim':o,'lr':lr,'decay':decay}
                    print('-'*80)
                    print(score,':',results[score])


# In[ ]:


# if searching for model is the case
# best_model = None
# best_search = None

# for model in results.keys():
#     print('* ',model,':',results[model]['AUCROC'])
#     if best_search is None or best_search['AUCROC'] < results[model]['AUCROC']:
#         best_search = results[model]
#         best_model = model


# In[ ]:


results


# In[ ]:


import collections


ordered_keys = list(sorted(results.keys()))

print(results[ordered_keys[0]])
best_optim = results[ordered_keys[0]]['optim']
best_lr = results[ordered_keys[0]]['lr']
best_activation = results[ordered_keys[0]]['activation']
best_decay = results[ordered_keys[0]]['decay']
best_size=results[ordered_keys[0]]['size']
# {'activation': 'nn.Hardtanh', 'optim': 'optim.Adamax', 'lr': 1, 'decay': 1e-05}


# In[ ]:


if best_activation in requeres_dim:
    activation = eval(best_activation)(dim=1)
else:
    activation = eval(best_activation)()
model = OverfitModel(activation,size=best_size)
print(model)
score,stats,search = one_cv(model,eval(best_optim)(model.parameters(),lr=best_lr,weight_decay=best_decay))


# In[ ]:


def plot(loss,val_loss,aucroc,val_aucroc):
    f,ax = plt.subplots(1,2)
    f.set_size_inches(14,6)
    ax[0].plot(loss, label='loss')
    ax[0].plot(val_loss, label='val_loss')
    ax[0].legend()
    ax[1].plot(aucroc, label ='aucroc')
    ax[1].plot(val_aucroc,label='val_aucroc')
    ax[1].legend()
    plt.plot()
    
    
for i in range(len(stats.split_losses)):
    plot(stats.split_losses[i],stats.split_val_losses[i],stats.split_aucrocs[i],stats.split_val_aucrocs[i])


# In[ ]:


submission_loader = create_submission_loader(test_x)
y_pred = eval_sub(model,submission_loader)
print(search)

df_subm = pd.DataFrame()
df_subm['id'] = df_test.id
df_subm['target'] = (y_pred > search['threshold']).astype(int)
print(df_subm.head())
print(df_subm.shape)
df_subm.to_csv('submission.csv', index=False)

