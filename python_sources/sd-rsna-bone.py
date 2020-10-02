#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import os
from sklearn.metrics import f1_score

from fastai import *
from fastai.vision import *

import torch
import torch.nn as nn
import torchvision
import cv2

from tqdm import tqdm
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


pd.read_csv("../input/boneage-training-dataset.csv").head(20)


# In[ ]:


pd.read_csv("../input/boneage-test-dataset.csv").head(20)


# In[ ]:


print(len(os.listdir("../input/boneage-training-dataset/boneage-training-dataset/")))


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


import torch
import torch.nn as nn
import torchvision
import cv2


# In[ ]:


model_path='.'
path='../input/'
train_folder=f'{path}boneage-training-dataset/boneage-training-dataset'
test_folder=f'{path}boneage-test-dataset/boneage-test-dataset'
train_lbl=f'{path}boneage-training-dataset.csv'
ORG_SIZE=96


# In[ ]:



bs=64
num_workers=None # Apprently 2 cpus per kaggle node, so 4 threads I think
sz=96


# In[ ]:


df_trn=pd.read_csv(train_lbl)


# In[ ]:


tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=.0, max_zoom=.1,
                      max_lighting=0.05, max_warp=0.)


# In[ ]:



data = ImageDataBunch.from_csv(path,csv_labels=train_lbl,folder='boneage-training-dataset/boneage-training-dataset', ds_tfms=tfms, size=sz, suffix='.png',test=test_folder,bs=bs);
stats=data.batch_stats()        
data.normalize(stats)


# In[ ]:


data.show_batch(rows=5, figsize=(12,9))


# In[ ]:


from torchvision.models import *


# In[ ]:



from sklearn.metrics import roc_auc_score
def auc_score(y_pred,y_true,tens=True):
    score=roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])
    if tens:
        score=tensor(score)
    else:
        score=score
    return score


# In[ ]:


learn = create_cnn(
    data,
    densenet201,
    path='.',    
    metrics=[auc_score], 
    ps=0.5
)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 3e-2


# In[ ]:


learn.fit_one_cycle(1,lr)
learn.recorder.plot()
learn.recorder.plot_losses()


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'fit_one_cycle')


# In[ ]:




