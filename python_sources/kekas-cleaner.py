#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -q albumentations')
get_ipython().system('pip install -q pretrainedmodels')
get_ipython().system('pip install -q kekas')
get_ipython().system('pip install -q adabound')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import pathlib
import time 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

import kekas

train_on_gpu = True
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from albumentations import Compose, JpegCompression, CLAHE, RandomRotate90, Transpose, ShiftScaleRotate,         Blur, OpticalDistortion, GridDistortion, HueSaturationValue, Flip, VerticalFlip
from albumentations import torch as AT
import pretrainedmodels as pm
import adabound

from kekas import Keker, DataOwner, DataKek
from kekas.transformations import Transformer, to_torch, normalize
from kekas.metrics import accuracy
from kekas.modules import Flatten, AdaptiveConcatPool2d
from kekas.callbacks import Callback, Callbacks, DebuggerCallback
from kekas.utils import DotDict


import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.

sample_sub = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


train_folder = '../input/plates/plates/train/'
test_folder = '../input/plates/plates/test/'

cleaned = 'cleaned'
dirty = 'dirty'

train_clean = os.listdir(train_folder + cleaned)
train_dirty = os.listdir(train_folder + dirty)
test_files = os.listdir(test_folder)

train_df = pd.concat([pd.DataFrame(train_clean, columns=['id']), pd.DataFrame(train_dirty, columns=['id'])])
train_df.reset_index(drop=True, inplace=True)
test_df = pd.DataFrame(test_files, columns=['id'])

train_df['label'] = [1 if file_name.startswith('cleaned') else 0 for file_name in train_df['id']]
test_df['label'] = -1
print(train_df.head(3))

fig = plt.figure(figsize=(20,5))
for idx, img in enumerate(np.random.choice(train_df['id'], 10)):
    ax = fig.add_subplot(2, 10//2, idx+1, xticks=[], yticks=[])
    im = Image.open(train_folder + img.split('_')[0] + '/' + img)
    plt.imshow(im)
    lab = img
    ax.set_title(f'{lab}')


# In[ ]:


def reader_train_fn(i, row):
    if row['id'].split('_')[0] == cleaned:
        image = cv2.imread(train_folder + cleaned + '/' + row['id'])[:,:,::-1] # BGR -> RGB
    else:
        image = cv2.imread(train_folder + dirty + '/' + row['id'])[:,:,::-1] # BGR -> RGB
    label = torch.Tensor([row["label"]])
    return {"image": image, "label": label}

def reader_test_fn(i, row):
    image = cv2.imread(test_folder + row['id'])[:,:,::-1] # BGR -> RGB
    label = torch.Tensor([row["label"]])
    return {"image": image, "label": label}

def step_fn(model: torch.nn.Module,
            batch: torch.Tensor) -> torch.Tensor:
    inp = batch["image"]
    return model(inp)

plt.imshow(reader_train_fn(32, train_df.loc[32])['image']);


# In[ ]:


def augs(p=0.5):
    return Compose([
        CLAHE(),
        RandomRotate90(),
        Transpose(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
        Blur(blur_limit=3),
        OpticalDistortion(),
        GridDistortion(),
        HueSaturationValue()
    ], p=p)

def get_transforms(dataset_key, size, p):
    PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))
    AUGS = Transformer(dataset_key, lambda x: augs()(image=x)["image"])
    NRM_TFMS = transforms.Compose([
        Transformer(dataset_key, to_torch()),
        Transformer(dataset_key, normalize())
    ])
    train_tfms = transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])
    val_tfms = transforms.Compose([PRE_TFMS, NRM_TFMS])  # because we don't want to augment val set yet
    return train_tfms, val_tfms


# In[ ]:


class Net(nn.Module):
    def __init__(
            self,
            num_classes: int,
            p: float = 0.5,
            pooling_size: int = 2,
            last_conv_size: int = 2048,
            arch: str = "squeezenet1_1",
            pretrained: str = "imagenet") -> None:
        """A simple model to finetune.
        
        Args:
            num_classes: the number of target classes, the size of the last layer's output
            p: dropout probability
            pooling_size: the size of the result feature map after adaptive pooling layer
            last_conv_size: size of the flatten last backbone conv layer
            arch: the name of the architecture form pretrainedmodels
            pretrained: the mode for pretrained model from pretrainedmodels
        """
        super().__init__()
        net = pm.__dict__[arch](pretrained=pretrained)
        modules = list(net.children())[:-1]  # delete last layers: pooling and linear
        
        modules += [nn.Sequential(
            AdaptiveConcatPool2d(size=pooling_size),
            Flatten(),
            nn.BatchNorm1d(
                2 * pooling_size * pooling_size * last_conv_size
            ),
            nn.Dropout(p),
            nn.Linear(
                2 * pooling_size * pooling_size * last_conv_size, 
                num_classes
            )
        )]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        logits = self.net(x)
        return logits


# In[ ]:


train, valid = train_test_split(train_df, stratify=train_df.label, test_size=0.15)


# In[ ]:


train_tfms, val_tfms = get_transforms("image", 224, 0.5)

train_dk = DataKek(df=train, reader_fn=reader_train_fn, transforms=train_tfms)
val_dk = DataKek(df=valid, reader_fn=reader_train_fn, transforms=val_tfms)
test_dk = DataKek(df=test_df, reader_fn=reader_test_fn, transforms=val_tfms)

batch_size = 2
workers = 8

train_dl = DataLoader(train_dk, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True)
val_dl = DataLoader(val_dk, batch_size=batch_size, num_workers=workers, shuffle=False)
test_dl = DataLoader(test_dk, batch_size=30, num_workers=workers, shuffle=False)


# In[ ]:


dataowner = DataOwner(train_dl, val_dl, None)

BCEmodel = Net(num_classes=1)
CEmodel = Net(num_classes=2)

CEcriterion = nn.CrossEntropyLoss()
BCEcriterion = nn.BCEWithLogitsLoss()


# In[ ]:


model = Net(num_classes=1, last_conv_size=1000)

def bce_accuracy(target: torch.Tensor,
                 preds: torch.Tensor) -> float:
    thresh = 0.5
    target = target.cpu().detach().numpy()
    preds = (torch.sigmoid(preds).cpu().detach().numpy() > thresh).astype(int)
    return accuracy_score(target, preds)

BCE_keker = Keker(model=model,
              dataowner=dataowner,
              criterion=BCEcriterion,
              step_fn=step_fn,
              target_key="label",
              metrics={"accuracy": bce_accuracy},
              opt=adabound.AdaBound,
              opt_params={'final_lr':0.001})


# In[ ]:


logdir = 'logs'
lrlogdir = 'lrlogs'

# !rm -r lrlogs/*

# BCE_keker.kek_lr(final_lr=0.1, logdir=lrlogdir)
# BCE_keker.plot_kek_lr(logdir=lrlogdir)


# In[ ]:


BCE_keker.freeze(model_attr='net')
BCE_keker.kek(lr=0.0001,
              epochs=50,
#               sched=torch.optim.lr_scheduler.StepLR,
#               sched_params={"step_size":10, "gamma": 0.5},
                        cp_saver_params={
                          "savedir": "bce_sueeze",  
#                           "metric": "accuracy",
                          "n_best": 3,
                          "prefix": "bce_squeeze",
                          "mode": "min"},
                        logdir=logdir)


# In[ ]:


BCE_keker.plot_kek(logdir=logdir, step='epoch')


# In[ ]:


BCE_keker.load('bce_sueeze/bce_squeeze.best.h5')
preds = BCE_keker.predict_loader(test_dl)
test_df['label'] = (torch.sigmoid(torch.Tensor(preds)) > 0.5).numpy()


# In[ ]:


test_df['id'] = test_df['id'].apply (lambda x: x.split('.')[0])
test_df['label'] = test_df['label'].apply(lambda x: 'cleaned' if x==1 else 'dirty')


# In[ ]:


fig = plt.figure(figsize=(20,5))
ids = np.random.choice(100, 30)
for idx, row in enumerate(ids):
    ax = fig.add_subplot(3, 30//3, idx+1, xticks=[], yticks=[])
    im = Image.open(test_folder + test_df.loc[row]['id'] + '.jpg')
    plt.imshow(im)
    lab = test_df.loc[row]['label']
    ax.set_title(f'{lab}')


# In[ ]:


test_df.to_csv('sub.csv', index=False)


# In[ ]:




