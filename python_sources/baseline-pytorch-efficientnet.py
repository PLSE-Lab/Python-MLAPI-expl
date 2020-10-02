#!/usr/bin/env python
# coding: utf-8

# ### The code has been written in Pytorch and used EfficientNet for classification.
# ### Custom Data of size 300 x 300 has been used, for fast loading of images, can be found:
# https://www.kaggle.com/bitthal/resize-jpg-siimisic-melanoma-classification
# 
# ### For EDA, refer:
# https://www.kaggle.com/bitthal/eda-working-with-dicom-images

# In[ ]:


import numpy as np
import pandas as pd

import os

from PIL import Image
from sklearn.model_selection import train_test_split

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import transforms
import torch
import torchvision

get_ipython().system('pip install efficientnet_pytorch')
from efficientnet_pytorch import EfficientNet


# In[ ]:


os.listdir('/kaggle/input')


# In[ ]:


PATH = '/kaggle/input/resize-jpg-siimisic-melanoma-classification/300x300'


# In[ ]:


df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
print(df.shape)
df.head()


# In[ ]:


df = df.sort_values('target', ascending=False).iloc[:5000, :].reset_index()
df.groupby('target').count()['image_name']


# In[ ]:


print("Total number of Images:", df.shape[0])
print("Total number of Malignmat Images:", df[df.target == 1].shape[0])
print("Total number of Benign Images:", df[df.target == 0].shape[0])


# In[ ]:


train_df, val_df = train_test_split(df, stratify=df.target, test_size=0.10)
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)


# In[ ]:


def default_image_loader(path):
    return Image.open(path).convert('RGB')

class ImageDataset(Dataset):
    def __init__(self, data_path, df, transform):
        self.df = df
        self.loader = default_image_loader
        self.transform = transform
        self.dir = data_path

    def __getitem__(self, index):
        image_name = self.df.image_name[index]
        image = self.loader(os.path.join(self.dir, image_name+'.jpg'))
        image = self.transform(image)
        
        if self.dir.split('/')[-1] == 'train':
            label = self.df.target[index]
            return image, label
            
        return image, image_name
            
        
    
    def __len__(self):
        return self.df.shape[0]


# In[ ]:


train_transform = transforms.Compose([
                              transforms.Resize((300, 300)),
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomRotation(20),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])

val_transform = transforms.Compose([
                              transforms.Resize((300, 300)),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])


# In[ ]:


img_dir = os.path.join(PATH, 'train')

train_dataset = ImageDataset(img_dir, train_df, train_transform)
val_dataset = ImageDataset(img_dir, val_df, val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for i, l in train_loader:\n    print(l)\n    break')


# In[ ]:


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.resnetmodel = EfficientNet.from_pretrained('efficientnet-b3')
        
        self.fc = nn.Sequential(nn.Linear(1000, 512), nn.ReLU(),
                                  nn.Linear(512, 1))
#                                 , nn.Sigmoid())
        
    def forward(self, x):
        x = self.resnetmodel(x)
        return self.fc(x)


# In[ ]:


model = Model()


# In[ ]:


import math
import torch
from torch.optim.optimizer import Optimizer, required

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:                    
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


# In[ ]:


criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = RAdam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)


# In[ ]:


model.cuda()


# In[ ]:


for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    model.train()
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.cuda()
        labels = labels.cuda()
        out = model(images)
        labels = labels.unsqueeze(1).float()
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
#         _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        out = torch.sigmoid(out)
        correct += ((out > 0.6).int() == labels).sum().item()
    
    print("Epoch: {}, Loss: {}, Train Accuracy: {}".format(epoch, running_loss, round(correct/total, 4)))
    if epoch % 2 == 1:
        scheduler.step()
        
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.cuda()
            labels = labels.cuda()
            labels = labels.unsqueeze(1).float()
            
            out = model(images)
            loss = criterion(out.data, labels)
            
            running_loss += loss.item()

            total += labels.size(0)
            out = torch.sigmoid(out)
            correct += ((out > 0.6).int() == labels).sum().item()
            
    print("Epoch: {}, Loss: {}, Test Accuracy: {}\n".format(epoch, running_loss, round(correct/total, 4)))
    
print('Finished Training')


# In[ ]:


test_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
test_df = test_df.reset_index(drop=True)
test_df.shape


# In[ ]:


test_df.head()


# In[ ]:


img_dir = os.path.join(PATH, 'test')

test_dataset = ImageDataset(img_dir, test_df, val_transform)
test_loader = DataLoader(test_dataset, batch_size=64, num_workers=8)


# In[ ]:


for a, b in test_loader:
#     print(b)
    pass
    break


# In[ ]:


model.eval()
res_id = []
res_prob = []

with torch.no_grad():
    for images, ids in test_loader:
        images = images.cuda()

        out = model(images)
        predicted = torch.sigmoid(out)
        
        res_id += ids
        res_prob += predicted.cpu().numpy().tolist()
        


# In[ ]:


res_prob = [x[0] for x in res_prob]
sum(res_prob)


# In[ ]:





# In[ ]:


sub = pd.DataFrame({"image_name":res_id, "target":res_prob})
sub.shape


# In[ ]:


sub.head()


# In[ ]:


sub.to_csv('submission.csv', index=False)


# In[ ]:




