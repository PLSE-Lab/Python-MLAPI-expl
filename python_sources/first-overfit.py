#!/usr/bin/env python
# coding: utf-8

# Just for fun:)
# 
# Using knowledge gained by competing in quora incencere questions:
# 
# https://www.kaggle.com/xsakix/pytorch-bilstm-meta-v2
# 
# 
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

seed = 6017

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed)
print('Seeding done...')


# In[ ]:


df_train = pd.read_csv("../input/train.csv")
print(df_train.drop(columns=['id']).head())
# print(df_train.drop(columns=['id'])['target'].values == df_train.target.values)
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


print(np.min(train_x))
print(np.max(train_x))
print(train_x.shape[0])

# for i in range(train_x.shape[1]):
#     print(np.mean(train_x[:,i]),':',np.std(train_x[:,i]))
    
mu_x = np.mean(train_x,axis=0)
std_x = np.std(train_x,axis=0)
print(mu_x.shape)


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


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchtext.data
import warnings
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.c1 = nn.Conv1d(300,256,kernel_size=1)
        self.generator = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,300),
            nn.Tanh()
        );

    def forward(self,x):
        x = x.view(x.shape[0],x.shape[1],1)
        x = F.relu(self.c1(x)).squeeze(2)
        return self.generator(x)

    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(300,256),
            nn.ReLU(),
            nn.Linear(256,1)
        );

    def forward(self,x):
        return self.discriminator(x)

    
    

def train_generator_on_set(data_set):
    batch_size=data_set.shape[0]

    train_tensor = torch.tensor(data_set, dtype=torch.float32).cuda()
    print(train_tensor.shape)
    train_dataset = torch.utils.data.TensorDataset(train_tensor)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size)

    generator = Generator().cuda()
    discriminator = Discriminator().cuda()

    criterion = nn.BCEWithLogitsLoss().cuda()        
    optimizerG = optim.Adam(generator.parameters(),lr=1e-3,weight_decay=1e-5)
    optimizerD = optim.Adam(discriminator.parameters(),lr=1e-3,weight_decay=1e-5)

    G_losses = []
    D_losses = []

    num_epochs=100

    real_label = 1
    fake_label = 0

    for epoch in range(num_epochs):
        for i,x_batch in enumerate(list(iter(train_loader)),1):
        ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            label = torch.full((batch_size,), real_label).cuda()
            # Forward pass real batch through D
            output = discriminator(x_batch[0]).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, 300).cuda()
            # Generate fake image batch with G
            label = torch.full((batch_size,), fake_label).cuda()
            fake = generator(noise)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            # Since we just updated D, perform another forward pass of all-fake batch through D
            label = torch.full((batch_size,), real_label).cuda()
            output = discriminator(fake).view(-1)        
            # Calculate G's loss based on this output
            errG = criterion(output, label)        
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if epoch % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(train_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

    plt.plot(G_losses,label='generator')
    plt.plot(D_losses,label='discriminator')
    plt.legend()
    plt.show()
    return generator


# In[ ]:


train_data_set = df_train.drop(columns=['id'])
print(train_data_set.shape)
train_1 = train_data_set[train_data_set.target == 1].drop(columns=['target'])
train_0 = train_data_set[train_data_set.target == 0].drop(columns=['target'])
print(train_1.shape)
print(train_0.shape)


# In[ ]:


generator_1 = train_generator_on_set(train_1.values)


# In[ ]:


generator_0 = train_generator_on_set(train_0.values)


# In[ ]:


# generate new data set
generator_1.eval()
generator_0.eval()
sample_size=1000000
new_data_1 = generator_1(torch.randn(int(sample_size/2), 300).cuda()).detach().cpu().numpy()
y_1 = np.full(int(sample_size/2),1)
new_data_0 = generator_0(torch.randn(int(sample_size/2), 300).cuda()).detach().cpu().numpy()
y_0 = np.full(int(sample_size/2),0)

new_x = np.concatenate([new_data_1,new_data_0, train_x])
new_y = np.concatenate([y_1,y_0,train_y])

print(new_x.shape)
print(new_y.shape)


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchtext.data
import warnings
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import warnings
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

batch_size=25

class OverfitModel(nn.Module):
    def __init__(self):
        super(OverfitModel,self).__init__()
        self.input_dim = 300
        self.hidden_dim=256
#         self.conv1 = nn.Conv1d(self.input_dim,self.hidden_dim,kernel_size=1)
#         self.drop_conv = nn.Dropout(0.1)
#         self.batch_norm_conv = nn.BatchNorm1d(self.input_dim)
#         self.conv2 = nn.Conv1d(self.hidden_dim,self.hidden_dim,kernel_size=1)
#         self.conv3 = nn.Conv1d(self.hidden_dim,self.hidden_dim,kernel_size=1)
#         self.mem = nn.LSTM(int(self.input_dim/50),self.hidden_dim,bidirectional=True,batch_first=True)
        self.classifier = nn.Sequential(
#             nn.BatchNorm1d(self.input_dim), - bad
            nn.Dropout(0.7),
            nn.Linear(self.input_dim,512),
#             nn.LayerNorm(256),
            nn.ReLU(True),
            nn.Dropout(0.9),
            nn.Linear(512,1)
        );

    def forward(self,x):
#         h = (torch.zeros(2,x.shape[0],self.hidden_dim).cuda(),torch.zeros(2,x.shape[0],self.hidden_dim).cuda())
#         x = x.view(x.shape[0],50,int(x.shape[1]/50))
#         _,h = self.mem(x)
#         x = torch.cat([h[0][-1,:,:],h[0][-2,:,:]],dim=1).cuda()
#         x = self.batch_norm_conv(x)
    
#         x = x.view(x.shape[0],x.shape[1],1)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv1(x)).squeeze(2)
#         print(x.shape)
        return self.classifier(x)


def eval_on_set(model,test_loader,loss_function):
    pred = []
    avg_loss = 0.
    with torch.no_grad():
        model.eval()
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
    with torch.no_grad():
        model.eval()
        for (x,) in list(submission_loader):       
            y_pred = torch.sigmoid(model(x).squeeze(1)).detach()
            pred += y_pred.cpu().numpy().tolist()

    return np.array(pred)

    
submission_dataset = torch.utils.data.TensorDataset(torch.tensor(test_x, dtype=torch.float32).cuda())
submission_loader = torch.utils.data.DataLoader(dataset=submission_dataset,batch_size=batch_size, shuffle=False)

# weights = compute_class_weight('balanced',np.unique(train_y),train_y)
# print(weights)
X_train1,X_val,y_train1, y_val = train_test_split(new_x,new_y,random_state=seed,stratify=new_y)
    
x_train_tensor = torch.tensor(X_train1, dtype=torch.float32).cuda()
y_train_tensor = torch.tensor(y_train1, dtype=torch.float32).cuda()
train_dataset = torch.utils.data.TensorDataset(x_train_tensor,y_train_tensor)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

x_val_tensor = torch.tensor(X_val, dtype=torch.float32).cuda()
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).cuda()
val_dataset = torch.utils.data.TensorDataset(x_val_tensor,y_val_tensor)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)

model = OverfitModel().cuda()

loss_function = nn.BCEWithLogitsLoss().cuda()        
optimizer = optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
early_stop = EarlyStopping(patience=5)

aurocs = []
val_aurocs = []
losses = []
val_losses = []

for epoch in range(20):
    y_pred,loss = train(model, train_loader,optimizer,loss_function)
    losses.append(loss)
    search = threshold_search(y_train1,y_pred)
    auroc = search['AUCROC']
    aurocs.append(auroc)
    y_pred, val_loss = eval_on_set(model,val_loader, loss_function)
    val_losses.append(val_loss)
    search = threshold_search(y_val,y_pred)
    val_aurocs.append(search['AUCROC'])
    print('EPOCH: ',epoch,': loss :',loss,': auroc: ',auroc,' : val loss: ',val_loss,': val AUC: ',search['AUCROC'])
    print('-'*80)
#     early_stop(np.round(1.-search['AUCROC'],decimals=4),model)
#     if early_stop.early_stop:
#         break

print('FINISHED TRAINING META...')
torch.save(model.state_dict(), 'checkpoint.pt')


# In[ ]:


f,ax = plt.subplots(1,2)
f.set_size_inches(18.5, 10.5)
ax[0].plot(aurocs,label='aurocs')
ax[0].plot(val_aurocs,label='val_aurocs')
ax[0].legend()
ax[1].plot(losses,label='losses')
ax[1].plot(val_losses,label='val_losses')
ax[1].legend()
plt.show()


# In[ ]:


x_train_tensor = torch.tensor(train_x, dtype=torch.float32).cuda()    
model = OverfitModel().cuda()
model.load_state_dict(torch.load('checkpoint.pt'))

#0.31
model.eval()
y_pred = torch.sigmoid(model(x_train_tensor)).squeeze(1)
y_pred = y_pred.detach().cpu().numpy()
print(y_pred.shape)

search_result = threshold_search(train_y, y_pred)
print(search_result)


# In[ ]:


y_pred = eval_sub(model,submission_loader)

df_subm = pd.DataFrame()
df_subm['id'] = df_test.id
df_subm['target'] = (y_pred > search_result['threshold']).astype(int)
print(df_subm.head())
print(df_subm.shape)
df_subm.to_csv('submission.csv', index=False)

