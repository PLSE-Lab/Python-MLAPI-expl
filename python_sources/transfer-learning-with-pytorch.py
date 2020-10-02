#!/usr/bin/env python
# coding: utf-8

# ### Transfer Learning with Pytorch

# In[ ]:


import os
import shutil
import numpy as np 
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


# In[ ]:


# train folders
os.mkdir('/kaggle/working/train/')
for img_class in os.listdir("/kaggle/input/animals10/raw-img"):
    os.mkdir('/kaggle/working/train/' + img_class + '/')


# In[ ]:


# test folders
os.mkdir('/kaggle/working/test/')
for img_class in os.listdir("/kaggle/input/animals10/raw-img"):
    os.mkdir('/kaggle/working/test/' + img_class + '/')


# In[ ]:


# form train dataset
for img_class in tqdm(os.listdir('/kaggle/working/train/')):
    img_ls = os.listdir('/kaggle/input/animals10/raw-img/' + img_class)
    for img in img_ls[:int(len(img_ls) * 0.8)]:

        shutil.copy('/kaggle/input/animals10/raw-img/' + img_class + '/' + img, 
                    '/kaggle/working/train/' + img_class + '/' + img)


# In[ ]:


# form test dataset
for img_class in tqdm(os.listdir('/kaggle/working/test/')):
    img_ls = os.listdir('/kaggle/input/animals10/raw-img/' + img_class)
    for img in img_ls[int(len(img_ls) * 0.8):]:

        shutil.copy('/kaggle/input/animals10/raw-img/' + img_class + '/' + img, 
                    '/kaggle/working/test/' + img_class + '/' + img)


# In[ ]:


train_data_path = "/kaggle/working/train/"
test_data_path = "/kaggle/working/test/"

transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225] )
    ])

# this function get folder with images
train_data = torchvision.datasets.ImageFolder(root=train_data_path,
                                              transform=transforms)

test_data = torchvision.datasets.ImageFolder(root=test_data_path,
                                             transform=transforms)


# In[ ]:


batch_size=64
train_data_loader = data.DataLoader(train_data, shuffle=True,
                                    batch_size=batch_size)

test_data_loader  = data.DataLoader(test_data, shuffle=True, 
                                    batch_size=batch_size)


# In[ ]:


from torchvision import models
# download pretrained ResNet
transfer_model = models.resnet50(pretrained=True)

# freeze all layers in pretrained model
for name, param in transfer_model.named_parameters():
    param.requires_grad = False
    
# replace last layer 
transfer_model.fc = nn.Sequential(nn.Linear(transfer_model.fc.in_features,500),
nn.ReLU(),
nn.Dropout(), nn.Linear(500,10))

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# In[ ]:


model = transfer_model 
model.to(device)
# add optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0003, amsgrad=True)
epochs = 50
loss_fn = torch.nn.CrossEntropyLoss()
loss_lst, loss_val_lst = [], []

for epoch in range(epochs):
    training_loss = 0.0
    valid_loss = 0.0
    model.train()
    for batch in train_data_loader:
        optimizer.zero_grad()
        inputs, target = batch
        inputs = inputs.to(device)
        target = target.to(device)
        output = model(inputs)
        
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        training_loss += loss.data.item()
    loss_lst.append(training_loss)

    model.eval()
    num_correct = 0
    num_examples = 0
    for batch in test_data_loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        output = model(inputs)
        targets = targets.to(device)
        loss = loss_fn(output,targets)
        valid_loss += loss.data.item()
        correct = torch.eq(torch.max(F.softmax(output), dim=1)[1],
                        targets).view(-1)
        num_correct += torch.sum(correct).item()
        num_examples += correct.shape[0]
    loss_val_lst.append(valid_loss)

    print('Epoch: {}, Training Loss: {:.2f},         Validation Loss: {:.2f},         accuracy = {:.2f}'.format(epoch, training_loss,         valid_loss, num_correct / num_examples))


# In[ ]:


plt.plot(range(len(loss_lst)), loss_lst);
plt.plot(range(len(loss_val_lst)), loss_val_lst);
plt.xlabel('epochs');
plt.ylabel('loss');


# ### Confusion matrix

# In[ ]:


all_output = np.array([])
all_targets = np.array([])

for batch in tqdm(test_data_loader):
    inputs, targets = batch
    inputs = inputs.to(device)
    output = model(inputs)
    all_output = np.concatenate([all_output, output.max(dim=1).indices.cpu().numpy()])
    all_targets = np.concatenate([all_targets, targets.numpy()])


# In[ ]:


# plot confusion matrix
conf_matr = confusion_matrix(all_output, 
                             all_targets)
plt.figure(figsize = (10,7))
sns.heatmap(conf_matr, annot=True);

