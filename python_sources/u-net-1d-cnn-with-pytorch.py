#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# * Ref Kernel https://www.kaggle.com/kmat2019/u-net-1d-cnn-with-keras, many thanks @K_mat shared 1DCNN keras version
# * Write the simple pytorch version for U-Net (1D CNN)

# ## Import Library

# In[ ]:


import os
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import cohen_kappa_score,f1_score
from sklearn.model_selection import KFold, train_test_split
from keras.callbacks import Callback
device = torch.device("cuda")


# ## Load and Split Dataset
# Simply split the input data into certain length.

# In[ ]:


df_train = pd.read_csv("../input/liverpool-ion-switching/train.csv")
df_test = pd.read_csv("../input/liverpool-ion-switching/test.csv")

# I don't use "time" feature
train_input = df_train["signal"].values.reshape(-1,4000,1)#number_of_data:1250 x time_step:4000
train_input_mean = train_input.mean()
train_input_sigma = train_input.std()
train_input = (train_input-train_input_mean)/train_input_sigma
test_input = df_test["signal"].values.reshape(-1,10000,1)
test_input = (test_input-train_input_mean)/train_input_sigma

train_target = pd.get_dummies(df_train["open_channels"]).values.reshape(-1,4000,11)#classification

idx = np.arange(train_input.shape[0])
train_idx, val_idx = train_test_split(idx, random_state = 111,test_size = 0.2)

val_input = train_input[val_idx]
train_input = train_input[train_idx] 
val_target = train_target[val_idx]
train_target = train_target[train_idx] 

print("train_input:{}, val_input:{}, train_target:{}, val_target:{}".format(train_input.shape, val_input.shape, train_target.shape, val_target.shape))


# ## Define Model
# This section defines U-Net(se-resnet base).
# Input and output of the U-Net are follows:
# * Input: 4000 time steps of "signal"
# * Output: (4000,11) time steps of "open_channels"

# In[ ]:


class conbr_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation = dilation, padding = 3, bias=True)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)
        
        return out       

class se_block(nn.Module):
    def __init__(self,in_layer, out_layer):
        super(se_block, self).__init__()
        
        self.conv1 = nn.Conv1d(in_layer, out_layer//8, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_layer//8, in_layer, kernel_size=1, padding=0)
        self.fc = nn.Linear(1,out_layer//8)
        self.fc2 = nn.Linear(out_layer//8,out_layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):

        x_se = nn.functional.adaptive_avg_pool1d(x,1)
        x_se = self.conv1(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv2(x_se)
        x_se = self.sigmoid(x_se)
        
        x_out = torch.add(x, x_se)
        return x_out

class re_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, dilation):
        super(re_block, self).__init__()
        
        self.cbr1 = conbr_block(in_layer,out_layer, kernel_size, 1, dilation)
        self.cbr2 = conbr_block(out_layer,out_layer, kernel_size, 1, dilation)
        self.seblock = se_block(out_layer, out_layer)
    
    def forward(self,x):

        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        x_re = self.seblock(x_re)
        x_out = torch.add(x, x_re)
        return x_out          

class UNET_1D(nn.Module):
    def __init__(self ,input_dim,layer_n,kernel_size,depth):
        super(UNET_1D, self).__init__()
        self.input_dim = input_dim
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth
        
        self.AvgPool1D1 = nn.AvgPool1d(input_dim, stride=5)
        self.AvgPool1D2 = nn.AvgPool1d(input_dim, stride=25)
        self.AvgPool1D3 = nn.AvgPool1d(input_dim, stride=125)
        
        self.layer1 = self.down_layer(self.input_dim, self.layer_n, self.kernel_size,1, 2)
        self.layer2 = self.down_layer(self.layer_n, int(self.layer_n*2), self.kernel_size,5, 2)
        self.layer3 = self.down_layer(int(self.layer_n*2)+int(self.input_dim), int(self.layer_n*3), self.kernel_size,5, 2)
        self.layer4 = self.down_layer(int(self.layer_n*3)+int(self.input_dim), int(self.layer_n*4), self.kernel_size,5, 2)
        self.layer5 = self.down_layer(int(self.layer_n*4)+int(self.input_dim), int(self.layer_n*5), self.kernel_size,4, 2)

        self.cbr_up1 = conbr_block(int(self.layer_n*7), int(self.layer_n*3), self.kernel_size, 1, 1)
        self.cbr_up2 = conbr_block(int(self.layer_n*5), int(self.layer_n*2), self.kernel_size, 1, 1)
        self.cbr_up3 = conbr_block(int(self.layer_n*3), self.layer_n, self.kernel_size, 1, 1)
        self.upsample = nn.Upsample(scale_factor=5, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=5, mode='nearest')
        
        self.outcov = nn.Conv1d(self.layer_n, 11, kernel_size=self.kernel_size, stride=1,padding = 3)
    
        
    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        for i in range(depth):
            block.append(re_block(out_layer,out_layer,kernel,1))
        return nn.Sequential(*block)
            
    def forward(self, x):
        
        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)
        pool_x3 = self.AvgPool1D3(x)
        
        #############Encoder#####################
        
        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)
        
        x = torch.cat([out_1,pool_x1],1)
        out_2 = self.layer3(x)
        
        x = torch.cat([out_2,pool_x2],1)
        x = self.layer4(x)
        
        #############Decoder####################
        
        up = self.upsample1(x)
        up = torch.cat([up,out_2],1)
        up = self.cbr_up1(up)
        
        up = self.upsample(up)
        up = torch.cat([up,out_1],1)
        up = self.cbr_up2(up)
        
        up = self.upsample(up)
        up = torch.cat([up,out_0],1)
        up = self.cbr_up3(up)
        
        out = self.outcov(up)
        
        #out = nn.functional.softmax(out,dim=2)
        
        return out


# ## Define Dataset with augmentation

# In[ ]:


class ION_Dataset(Dataset):
    def __init__(self, train_input, train_output,mode='train'):
        self.train_input = train_input
        self.train_output = train_output
        self.mode = mode
        
    def __len__(self):
        return len(self.train_input)
    
    def _augmentations(self,input_data, target_data):
        #flip
        if np.random.rand()<0.5:    
            input_data = input_data[::-1]
            target_data = target_data[::-1]
        return input_data, target_data
    
    def __getitem__(self, idx):
        x = self.train_input[idx]
        y = self.train_output[idx]
        if self.mode =='train':
            x,y = self._augmentations(x,y)
        out_x = torch.tensor(np.transpose(x.copy(),(1,0)), dtype=torch.float)
        out_y = torch.tensor(np.transpose(y.copy(),(1,0)), dtype=torch.float)
        return out_x, out_y


# In[ ]:


batch_size = 8
train = ION_Dataset(train_input, train_target,mode='train')
valid = ION_Dataset(val_input, val_target,mode='valid')

x_test = torch.tensor(np.transpose(test_input,(0,2,1)), dtype=torch.float).cuda()
test = torch.utils.data.TensorDataset(x_test)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)


# In[ ]:


x_train, y = next(iter(train_loader))
x_train.shape, y.shape


# ## Training

# In[ ]:


from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR,ReduceLROnPlateau
import time
from tqdm import tqdm
## Hyperparameter
n_epochs = 100
lr = 0.001

## Build tensor data for torch
train_preds = np.zeros((int(train_input.shape[0]*train_input.shape[1])))
val_preds = np.zeros((int(val_input.shape[0]*val_input.shape[1])))

train_targets = np.zeros((int(train_input.shape[0]*train_input.shape[1])))

avg_losses_f = []
avg_val_losses_f = []

##Loss function
loss_fn = torch.nn.BCEWithLogitsLoss()

#Build model, initial weight and optimizer
model = UNET_1D(1,128,7,3) #(input_dim, hidden_layer, kernel_size, depth)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr,weight_decay=1e-5) # Using Adam optimizer
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.8, min_lr=1e-8) # Using ReduceLROnPlateau schedule
temp_val_loss = 9999999999


for epoch in range(n_epochs):
    
    start_time = time.time()
    model.train()
    avg_loss = 0.
    for i, (x_batch, y_batch) in enumerate(train_loader):
        y_pred = model(x_batch.cuda())
        
        loss = loss_fn(y_pred.cpu(), y_batch)
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        avg_loss += loss.item()/len(train_loader)

        pred = F.softmax(y_pred, 1).detach().cpu().numpy().argmax(axis=1)
        train_preds[i * batch_size*train_input.shape[1]:(i+1) * batch_size*train_input.shape[1]] = pred.reshape((-1))
        train_targets[i * batch_size*train_input.shape[1]:(i+1) * batch_size*train_input.shape[1]] = y_batch.detach().cpu().argmax(axis=1).reshape((-1))
        del y_pred, loss, x_batch, y_batch, pred
        
        
    model.eval()

    avg_val_loss = 0.
    for i, (x_batch, y_batch) in enumerate(valid_loader):
        y_pred = model(x_batch.cuda()).detach()

        avg_val_loss += loss_fn(y_pred.cpu(), y_batch).item() / len(valid_loader)
        pred = F.softmax(y_pred, 1).detach().cpu().numpy().argmax(axis=1)
        val_preds[i * batch_size*val_input.shape[1]:(i+1) * batch_size*val_input.shape[1]] = pred.reshape((-1))
        del y_pred, x_batch, y_batch, pred
        
    if avg_val_loss<temp_val_loss:
        #print ('checkpoint_save')
        temp_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'ION_train_checkpoint.pt')
        
    train_score = f1_score(train_targets,train_preds,average = 'macro')
    val_score = f1_score(val_target.argmax(axis=2).reshape((-1)),val_preds,average = 'macro')
    
    elapsed_time = time.time() - start_time 
    scheduler.step(avg_val_loss)
    
    print('Epoch {}/{} \t loss={:.4f} \t train_f1={:.4f} \t val_loss={:.4f} \t val_f1={:.4f} \t time={:.2f}s'.format(
        epoch + 1, n_epochs, avg_loss,train_score, avg_val_loss,val_score, elapsed_time))


# In[ ]:


print("VALIDATION_SCORE (QWK): ", cohen_kappa_score(val_target.argmax(axis=2).reshape((-1)),val_preds, weights="quadratic"))


# In[ ]:


print("VALIDATION_SCORE (F1): ", f1_score(val_target.argmax(axis=2).reshape((-1)),val_preds ,average = 'macro'))


# ## Predict and Submit
# This is not the main topic of this kernel, so I just round predicted values.

# In[ ]:


model.load_state_dict(torch.load('ION_train_checkpoint.pt'))
model.eval()
test_preds = np.zeros((int(test_input.shape[0]*test_input.shape[1])))
for i, x_batch in enumerate(test_loader):
    y_pred = model(x_batch[0]).detach()

    pred = F.softmax(y_pred, 1).detach().cpu().numpy().argmax(axis=1)
    test_preds[i * batch_size*test_input.shape[1]:(i+1) * batch_size*test_input.shape[1]] = pred.reshape((-1))
    del y_pred, x_batch, pred


# In[ ]:


df_sub = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv", dtype={'time':str})
df_sub.open_channels = np.array(test_preds,np.int)
df_sub.to_csv("submission.csv",index=False)


# In[ ]:


df_sub.head()


# ## Next plan
# * K-fold split training
# * Change loss function
# * Try RNN or LSTM model
