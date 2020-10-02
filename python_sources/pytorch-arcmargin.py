#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F

import torchvision
from torchvision import transforms as T

import os, time, random, sys, math


# In[ ]:


df_train = pd.read_csv('../input/Kannada-MNIST/train.csv')
df_test = pd.read_csv('../input/Kannada-MNIST/test.csv')


# In[ ]:


class Identity():
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    """
    def __init__(self, *args, **kwargs):
        pass
    
    def __repr__(self):
        format_string = self.__class__.__name__ 
        return format_string
    
    def __call__(self, input):
        return input
    
class DataSet(D.Dataset):
    def __init__(self, df, transform=None, mode='train'):
        self.mode = mode
        self.transform = transform if transform else Identity()
        self.len = len(df)
        self.images = df.iloc[:,1:].values.reshape(-1,28,28,1)
        if mode == 'train':
            self.y = df.label.values
    
    def __getitem__(self, index):
        
        image = self.images[index]
        image = self.transform(image)
        if self.mode == 'train':
            return image, self.y[index]
        else:
            return image
            
    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
    
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=15.0, m=0.35, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if not self.training:
            return cosine * self.s
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output


# In[ ]:


batch_size = 4*1024
device = 'cuda'

trfm = T.Compose([
        T.Lambda(lambda x: x / 255),
        T.ToTensor(),
        T.Lambda(lambda x: x.float()),
    ])

ds = DataSet(df_train[:50000], transform=trfm)
loader = D.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

vds = DataSet(df_train[50000:], transform=trfm)
vloader = D.DataLoader(vds, batch_size=batch_size, shuffle=False, num_workers=2)


# In[ ]:


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # image starts as (1, 28, 28)
        # Formula to compute size of image after conv/pool
        # (size-filter+2*padding / stride) + 1
        #                      inputs         # of filters    filter size    
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2) # conv1
        self.conv1_bn = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2) # conv2
        self.conv2_bn = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels= 128, kernel_size=3, stride=1, padding=1) # conv3
        self.conv3_bn = nn.BatchNorm2d(num_features=128)
        
        self.fc1 = nn.Linear(in_features=128*6*6, out_features=1024) # linear 1
        self.fc1_bn = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512) # linear 2
        self.fc2_bn = nn.BatchNorm1d(num_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=256) # linear 3
        self.fc3_bn = nn.BatchNorm1d(num_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=64) # linear 4
        self.fc4_bn = nn.BatchNorm1d(num_features=64)
        self.arc = ArcMarginProduct(64,10,s=17,m=0.5) # output
    
    def forward(self, t, y=None):
        t = F.relu(self.conv1_bn(self.conv1(t)))
        t = F.max_pool2d(t, kernel_size=2, stride=2) # (1, 14, 14)
        
        t = F.relu(self.conv2_bn(self.conv2(t)))
        t = F.max_pool2d(t, kernel_size=2, stride=2) # (1, 7, 7)
        
        t = F.relu(self.conv3_bn(self.conv3(t)))
        t = F.max_pool2d(t, kernel_size=2, stride=1) # (1, 6, 6)
        
        t = F.relu(self.fc1_bn(self.fc1(t.reshape(-1, 128*6*6))))
        t = F.relu(self.fc2_bn(self.fc2(t)))
        t = F.relu(self.fc3_bn(self.fc3(t)))
        t = F.relu(self.fc4_bn(self.fc4(t)))
        t = self.arc(t, y)
        
        return t


# In[ ]:


class Flatten(nn.Module):  
    def forward(self, x):
        return torch.flatten(x, start_dim=1, end_dim=-1)
    
class ArcModel(nn.Module):
    def __init__(self, s=17.0, m=0.5):
        super().__init__()
        self.conv = nn.Sequential(
                            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
                            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
                            nn.Dropout(0.4),
                            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2),
                            nn.Dropout(0.4),
                            Flatten(),
                            nn.Linear(64*7*7, 128),
                            nn.ReLU(inplace=True),
                            nn.BatchNorm1d(128),
                            nn.Dropout(0.4),
                        )
        
        self.arc = ArcMarginProduct(128,10,s=s,m=m)
        self.linear = nn.Linear(128, 10)
        
    def forward(self, x, y=None):
        x = self.conv(x)
#         x = self.arc(x.squeeze(-1).squeeze(-1), y)
        x =self.linear(x)
        return x


# In[ ]:


@torch.no_grad()
def validation(model, loader, loss_fn):
    model.eval()
    losses = []
    accs = []
    for x, y in loader: 
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = loss_fn(output, y)
        losses.append(loss.item())
        accs.append(accuracy(output.cpu(), y.cpu()))
    return np.array(losses), np.array(accs)

def accuracy(output, target, topk=(1,3)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return np.array(res)


# In[ ]:


model = ArcModel()


# In[ ]:


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),
                  lr=1e-3, betas=(0.9, 0.999), amsgrad=True)

# optimizer = torch.optim.SGD(model.parameters(),
#                   lr=1e-2, momentum=0.9)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [7, 70], 0.1)


# In[ ]:


model.to(device);


# In[ ]:


epochs = 100
for epoch in range(1, epochs+1):
    losses = []
    accs = []

    model.train()
    for x, y in loader: 
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        probs = model(x, y)
        loss = criterion(probs, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accs.append(accuracy(probs.cpu(), y.cpu()))

        del loss, probs, y, x
        
    scheduler.step()
    
    losses = np.array(losses).mean()
    accs = np.array(accs).mean(axis=0)

    vlosses, vaccs = validation(model, vloader, criterion)
    vlosses = vlosses.mean()
    vaccs = vaccs.mean(axis=0)

print('Epoch {:3d} -> Train Loss: {:6.3f}, ACC: {:5.2f}%, TOP-3: {:5.2f}%, Valid Loss: {:6.3f}, ACC: {:5.2f}%, TOP-3: {:5.2f}%'
    .format(epoch, losses, accs[0], accs[1], vlosses, vaccs[0], vaccs[1]))


# In[ ]:


tds = DataSet(df_test, transform=trfm, mode='test')
tloader = D.DataLoader(tds, batch_size=batch_size, shuffle=False, num_workers=2)


# In[ ]:


preds = []
model.eval()
for x in tloader: 
    x = x.to(device)
    probs = model(x)
    preds.append(probs.detach().cpu().numpy().argmax(axis=1))
    
preds = np.concatenate(preds)


# In[ ]:


submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
submission['label'] = preds
submission.to_csv('submission.csv', index=False)

