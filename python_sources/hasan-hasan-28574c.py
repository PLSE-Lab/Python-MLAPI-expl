#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from fastai.vision import *
from torchvision import models as mod
from torch import nn

norm_values=([0.485,0.456,0.406],[0.229,0.225,0.224])
iB = ImageDataBunch.from_folder(path='../input/flower_data/flower_data',
                                           size=224,bs=64,
                                ds_tfms=get_transforms(flip_vert=True,do_flip=True,max_rotate=90))
iB=iB.normalize(norm_values)
model = mod.resnext101_32x8d(pretrained=True)

for param in model.parameters():
    param.requires_grad=False
fc = nn.Sequential(        nn.Dropout(p=0.5),                        
                          nn.Linear(2048,1000),
                          nn.BatchNorm1d(1000),
                          nn.ReLU(),
                          nn.Linear(1000,102),
                          nn.LogSoftmax(dim=1))
model.fc=fc
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


# In[ ]:


learn = Learner(data=iB,model=model,model_dir='/tmp/model',metrics=[accuracy])
learn.lr_find()

learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(8,slice(2e-02))

