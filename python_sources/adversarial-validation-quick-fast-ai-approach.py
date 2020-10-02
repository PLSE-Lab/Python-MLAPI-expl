#!/usr/bin/env python
# coding: utf-8

# Several people have reported a discrepancy between CV and LB scores. The main idea behind this kernel is to have a quick and dirty check: how different are the distributions of the classes between training and test sets? The approach I use is adversarial validation:
# 
# http://fastml.com/adversarial-validation-part-one/
# 
# 

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.metrics import cohen_kappa_score

import numpy as np
import scipy as sp
from functools import partial
from sklearn import metrics
from collections import Counter
import json

from PIL import Image


import time
import torchvision
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm

from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
import os

device = torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


# In[ ]:


# settings
bs = 64 
sz = 224


# In[ ]:


# # Making pretrained weights work without needing to find the default filename
if not os.path.exists('/tmp/.cache/torch/checkpoints/'):
        os.makedirs('/tmp/.cache/torch/checkpoints/')
get_ipython().system("cp '../input/resnet50/resnet50.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'")


# # Data

# The point of this block is to combine the training and test data into a single data frame, which can subsequently be used in our pipeline.

# In[ ]:


# training images
base_image_dir = os.path.join('..', 'input/aptos2019-blindness-detection/')
train_dir = os.path.join(base_image_dir,'train_images/')
df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))
df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))
df = df.drop(columns=['id_code'])
df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
df['is_test'] = 0
df.drop('diagnosis', axis = 1, inplace = True)

df1 = df.copy()


# In[ ]:


# test images
base_image_dir = os.path.join('..', 'input/aptos2019-blindness-detection/')
train_dir = os.path.join(base_image_dir,'test_images/')
df = pd.read_csv(os.path.join(base_image_dir, 'test.csv'))
df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))
df = df.drop(columns=['id_code'])
df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
df['is_test'] = 1
df2 = df.copy()


# In[ ]:


df_total = pd.concat([df1,df2], axis =0 )
df_total = df_total.sample(frac=1).reset_index(drop=True) 
del df1, df2


# In[ ]:


# add cv folds indices (yes, i know it's ugly :-)
kf = KFold(n_splits=5)

df_total['fold_id'] = -1

for (nf, (train_index, test_index)) in enumerate(kf.split(df_total)):
    df_total['fold_id'][test_index] = nf


# # Model

# Loop over folds - check performance for each

# In[ ]:


res = np.zeros((5,1))


# In[ ]:


for ii in range(0, 5):
    
    # create this split for training / validation 
    df = df_total.copy()
    df['is_valid'] = (df['fold_id'] == ii) + 0
    df.drop('fold_id', axis = 1, inplace = True)
    
    # create the data object
    tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,max_warp=0,max_zoom=1.1,max_lighting=0.1,p_lighting=0.5)
    src = (ImageList.from_df(df=df,path='./',cols='path') 
        .split_from_df() 
        .label_from_df(cols='is_test') 
      )
    data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='zeros')
        .databunch(bs=bs,num_workers=4)
        .normalize(imagenet_stats)   
       )
    
    # train a model for this fold - no optimization
    learn = cnn_learner(data, base_arch = models.resnet50)
    learn.unfreeze()
    learn.fit_one_cycle(1, max_lr = slice(1e-6,1e-3))
    
    # evaluate performance
    img = learn.data.valid_dl
    xpred = learn.get_preds(img)
    xscore = roc_auc_score(xpred[1],xpred[0][:,1])
    print('fold '+str(ii) + ': ' + str(np.round(xscore, 4)))

    res[ii] = xscore
    


# As can be seen from the results above (each fold has AUC > 0.9), even with a clearly underfitting model (validation loss < training loss) we can quite accurately distinguish the training and test sets. This means garden variety random split just won't do the job :-(

# In[ ]:


print(res)

