#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""
# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing,


# In[ ]:


from pathlib import Path
import torchvision.models as models
import pickle, gzip, math, torch, matplotlib as mpl
import matplotlib.pyplot as plt
from torch import tensor
import os


# In[ ]:



import torch
from torch import nn
import pathlib
from torch.utils.data import DataLoader
from torchvision import *


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


torch.cuda.empty_cache()


# In[ ]:


transformtrain= transforms.Compose([
    transforms.Resize((124,124)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),              #convert the value to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])         # convert all the value form -1 to 1 for all RGB
])


# In[ ]:


transformvalid= transforms.Compose([
    transforms.Resize((124,124)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #convert the value to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])         # convert all the value form -1 to 1 for all RGB
])


# In[ ]:



model =models.resnext50_32x4d(pretrained=True).to(device)


# In[ ]:


model


# In[ ]:



traindata=datasets.ImageFolder('/kaggle/input/intel-image-classification/seg_train/seg_train/' , transform=transformtrain)


# In[ ]:



valdata=datasets.ImageFolder('/kaggle/input/intel-image-classification/seg_test/seg_test/', transform=transformvalid)


# In[ ]:


trainloader=DataLoader(traindata, batch_size=64, shuffle=True)


# In[ ]:


valloader=DataLoader(valdata,batch_size=64,shuffle=True)


# In[ ]:


classes=['buildings','forest','glacier','mountain','sea','street']


# In[ ]:


for param in model.parameters():
    param.requires_grad=False


# In[ ]:



model.fc=nn.Linear(2048,6).to(device)

criterion=nn.CrossEntropyLoss()


# In[ ]:


optimizer=torch.optim.Adam(model.parameters(),lr=0.0004)


# In[ ]:


optimizer.zero_grad()


# In[ ]:


def accuracy(out, yb): 
    return (torch.argmax(out, dim=1)==yb).float().mean()


# In[ ]:



def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    tloss=[]
    vloss=[]
    
    for epoch in range(epochs):
        # Handle batchnorm / dropout
        model.train()
        tot_train=0
        model.train()
#         print(model.training)
        for xb,yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_func(model(xb), yb)
            tot_train+=loss
            loss.backward()
            opt.step()
            opt.zero_grad()
        nt=len(train_dl)
        
        
        
        model.eval()
#         print(model.training)
        with torch.no_grad():
            tot_loss,tot_acc = 0.,0.
            for xb,yb in valid_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                tot_loss += loss_func(pred, yb)
                tot_acc  += accuracy (pred,yb)
        nv = len(valid_dl)
        tloss.append(tot_train/nt)
        vloss.append(tot_loss/nv)
        print(epoch,tot_train/nt, tot_loss/nv, tot_acc/nv)
    return tloss, vloss


# In[ ]:



ltrain,lval = fit(10, model, criterion, optimizer, trainloader, valloader)


# In[ ]:


plt.plot(ltrain, label='Training loss', color='green')
plt.plot(lval, label='Validation loss', color ='black')
plt.legend(frameon=False)
plt.show()


# In[ ]:


PATH="/kaggle/working/save"
torch.save({
            
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            
            
            }, PATH)


# In[ ]:


checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
# modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

# modelA.eval()
# modelB.eval()
# # - or -
model.train()
# modelB.train()


# In[ ]:


for param in model.parameters():
    param.requires_grad = True


# In[ ]:


optimizer = torch.optim.Adam([{'params': model.layer4.parameters()},
                {'params': model.layer1.parameters(), 'lr': 5e-6},
                 {'params': model.layer2.parameters(), 'lr': 1e-5},
                {'params': model.layer3.parameters(), 'lr': 5e-5}
            ], lr=1e-4)


# In[ ]:


ltrain,lval = fit(6, model, criterion, optimizer, trainloader, valloader)


# In[ ]:



PATH="/kaggle/working/save2"
torch.save({
            
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            
            
            }, PATH)


# In[ ]:


checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
# modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

# modelA.eval()
# modelB.eval()
# # - or -

# modelB.train()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



plt.plot(ltrain, label='Training loss', color='green')
plt.plot(lval, label='Validation loss', color ='black')
plt.legend(frameon=False)
plt.show()


# In[ ]:


transformtrain= transforms.Compose([
    transforms.Resize((250,250)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),              #convert the value to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])         # convert all the value form -1 to 1 for all RGB
])


# In[ ]:



transformvalid= transforms.Compose([
    transforms.Resize((250,250)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #convert the value to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])         # convert all the value form -1 to 1 for all RGB
])


# In[ ]:



for param in model.parameters():
    param.requires_grad=True


# In[ ]:



optimizer=torch.optim.Adam(model.parameters(),lr=0.000001)

traindata=datasets.ImageFolder('/kaggle/input/intel-image-classification/seg_train/seg_train' , transform=transformtrain)

valdata=datasets.ImageFolder('/kaggle/input/intel-image-classification/seg_test/seg_test', transform=transformvalid)

trainloader=DataLoader(traindata, batch_size=32, shuffle=True)

valloader=DataLoader(valdata,batch_size=32,shuffle=True)


# In[ ]:


ltrain,lval = fit(6, model, criterion, optimizer, trainloader, valloader)


# In[ ]:


plt.plot(ltrain, label='Training loss', color='green')
plt.plot(lval, label='Validation loss', color ='black')
plt.legend(frameon=False)
plt.show()


# In[ ]:



PATH="/kaggle/working/saveprogresive"
torch.save({
            
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            
            
            }, PATH)


# In[ ]:



checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
# modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

# modelA.eval()
# modelB.eval()
# # - or -
model.train()


# In[ ]:



for param in model.parameters():
    param.requires_grad = True


# In[ ]:


optimizer = torch.optim.Adam([{'params': model.layer4.parameters()},
                {'params': model.layer1.parameters(), 'lr': 1e-6},
                 {'params': model.layer2.parameters(), 'lr': 1e-5},
                {'params': model.layer3.parameters(), 'lr': 5e-5}
            ], lr=1e-4)


# In[ ]:


ltrain,lval = fit(6, model, criterion, optimizer, trainloader, valloader)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




