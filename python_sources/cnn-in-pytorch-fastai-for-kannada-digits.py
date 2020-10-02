#!/usr/bin/env python
# coding: utf-8

# This notebook trains a cnn model on Kannada digits recognition with <b>pytorch</b>, <b>fastai</b>, and faster image transformation package <b>albumentations</b> rather than torchvision.

# In[ ]:


import os, time, pickle, random

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import fastai
from fastai.train import Learner
from fastai.train import DataBunch
from fastai.metrics import accuracy
from fastai.callbacks import *
from fastai.basic_data import DatasetType

from PIL import Image
from torchvision import models
import torchvision.datasets as dset
import torchvision.transforms as T

import albumentations as albu

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float32

img_size = 28
bs = 256
lr = 5e-2
epochs = 28


# ## Dataset

# In[ ]:


train = pd.read_csv('../input/Kannada-MNIST/train.csv')
dev = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')
test = pd.read_csv('../input/Kannada-MNIST/test.csv')

train.shape, dev.shape, test.shape


# In[ ]:


train_x, train_y = train.values[:,1:], train.values[:,0]
x_dev, y_dev = dev.values[:,1:], dev.values[:,0]
x_test = test.values[:,1:]

del train, dev, test


# In[ ]:


print((train_x.mean(), x_dev.mean(), x_test.mean()))
x_mean, x_std = np.concatenate([train_x, x_dev, x_test], 0).mean() / 255, np.concatenate([train_x, x_dev, x_test], 0).std() / 255
print(x_mean, x_std)


# Observations from images and model results show that data distribution is different in 3 given datasets train/test/Dig-MNIST.

# In[ ]:


# train images
fig, axes = plt.subplots(10, 10, figsize=(10,10))

for i in range(10):
    ims = train_x[train_y == i]
    axes[0][i].set_title(i)
    for j in range(10):
        axes[j][i].axis('off')
        axes[j][i].imshow(ims[j,:].reshape(28, 28), cmap='gray')


# In[ ]:


# Dig-MNIST images
fig, axes = plt.subplots(10, 10, figsize=(10,10))

for i in range(10):
    ims = x_dev[y_dev == i]
    axes[0][i].set_title(i)
    for j in range(10):
        axes[j][i].axis('off')
        axes[j][i].imshow(ims[j,:].reshape(28, 28), cmap='gray')


# In[ ]:


# test images
fig, axes = plt.subplots(5, 5, figsize=(8,8))

ims = x_test[np.random.choice(len(x_test), 25),:]
for i, im in enumerate(ims):
    ax = axes[i//5, i%5]
    ax.axis('off')
    ax.imshow(im.reshape(28,28), cmap='gray')


# In[ ]:


norm = {'mean': x_mean, 'std': x_std}

tsfm_aug = albu.Compose([
                         albu.Resize(img_size, img_size),
                         albu.RandomContrast(limit=0.5),
                         albu.RandomBrightness(limit=0.5),
                         albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=15, shift_limit=0.25, border_mode=0),
            ])

tsfm_normal = albu.Compose([
                            albu.Resize(img_size, img_size),
                            
            ])

class Kannada_mnist(Dataset):
    def __init__(self, x, y=None, transforms=tsfm_normal, label_smooth=0):
        assert 0. <= label_smooth <= 1.
        self.x = x
        self.y = y
        self.transforms = transforms
        self.label_smooth = label_smooth

    def __getitem__(self, idx):
        img = self.transforms(image=self.x[idx].astype('uint8'))['image'] / 255.
        img = (img - norm['mean']) / norm['std']
        img = img[None,:].astype('float32')
        if self.y is None:
            return img
        if self.label_smooth > 0:
            label = np.zeros(10)
            label[self.y[idx]] += 1
            label = label*(1-self.label_smooth) + self.label_smooth / 10
        else:
            label = self.y[idx]
        return img, label

    def __len__(self):
        return len(self.x)


# ## Model

# In[ ]:


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`,"
    "a module from fastai v1."
    def __init__(self, output_size=None):
        "Output will be 2*output_size or 2 if output_size is None"
        super().__init__()
        self.output_size = output_size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

def vgg_style():
    return nn.Sequential(
        nn.Conv2d(1, 64, 3, padding=1),
        nn.PReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.PReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.PReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.PReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(128, 256, 3, padding=1),
        nn.PReLU(),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.PReLU(),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.PReLU(),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.PReLU(),
        nn.BatchNorm2d(256),
        AdaptiveConcatPool2d((3, 3)),
        Flatten(),
        nn.Linear(256 * 9 * 2, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Dropout(p=0.2),
        nn.Linear(256, 10),
    )

def model_test():
    x = torch.zeros((64,1,28,28), dtype=dtype)
    model = vgg_style()
    model = model.to(device)
    print(model(x.to(device)).size())

model_test()


# ## Train the cnn

# In[ ]:


def seed_torch(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_preds(test_loader, model, dset_type='test'):
    model.eval()
    scores = []
    with torch.no_grad():
        for x in test_loader:
            if dset_type == 'val': x = x[0]
            x = x.to(device=device, dtype=dtype)
            score = model(x)
            scores.append(F.softmax(score, -1).cpu().numpy())
    scores = np.concatenate(scores)
    return scores


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=0)

train_dataset = Kannada_mnist(x_train.reshape(-1, 28, 28), y_train, transforms=tsfm_aug)
val_dataset = Kannada_mnist(x_val.reshape(-1, 28, 28), y_val)
dev_dataset = Kannada_mnist(x_dev.reshape(-1, 28, 28), y_dev)
test_dataset = Kannada_mnist(x_test.reshape(-1, 28, 28))

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
dev_loader = DataLoader(dev_dataset, batch_size=bs, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

databunch = DataBunch(train_dl=train_loader, valid_dl=val_loader)


# In[ ]:


seed_torch()
model = vgg_style()
learn = Learner(databunch, model, loss_func=F.cross_entropy, metrics=accuracy)
learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(epochs, max_lr=lr)

print()
print('GPU memory:')
print(str(torch.cuda.memory_allocated(device)/1e6 ) + 'M')
print(str(torch.cuda.memory_cached(device)/1e6 ) + 'M')
torch.cuda.empty_cache()


# In[ ]:


dev_preds = get_preds(dev_loader, model, 'val')
test_preds = get_preds(test_loader, model)
dev_preds = np.argmax(dev_preds, -1)
test_preds = np.argmax(test_preds, -1)
print('Dev. ACC: %.4f' %((dev_preds == dev_dataset.y).mean(),))


# In[ ]:


submission = pd.DataFrame({'id': np.arange(len(test_preds)), 'label': test_preds})
submission.to_csv('submission.csv', index=False)


# Comparison of predictions and true images

# In[ ]:


dev_preds[:9]


# In[ ]:


fig, axes = plt.subplots(3, 3, figsize=(8,8))

ims = x_dev[:9,:]
for i, im in enumerate(ims):
    ax = axes[i//3,i%3]
    ax.imshow(im.reshape(28,28), cmap='gray')


# In[ ]:


test_preds[:9]


# In[ ]:


fig, axes = plt.subplots(3, 3, figsize=(8,8))

ims = x_test[:9,:]
for i, im in enumerate(ims):
    ax = axes[i//3,i%3]
    ax.imshow(im.reshape(28,28), cmap='gray')

