#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai import *
from fastai.vision import *

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold

from math import floor


# In[ ]:


from glob import glob
imagePatches = glob('../input/kimia_path_960/*.*', recursive=True)


# In[ ]:


print(os.path.basename('../input/kimia_path_960/A5.tif').split(".")[0][0])


# In[ ]:


y = []
for img in imagePatches:
    label = os.path.basename(img).split(".")[0][0]
    y.append(label)


# In[ ]:


len(y)


# In[ ]:


images_df = pd.DataFrame()
images_df["name"] = imagePatches
images_df["label"] = y


# In[ ]:


skf = StratifiedKFold(n_splits=10)


# In[ ]:


path=""
tfms = get_transforms()
accuracy_list = []


# In[ ]:


for train_index, val_index in skf.split(images_df["name"], images_df["label"]):
    print(train_index,val_index)
    data = (ImageList.from_df(images_df,path)
        #Where to find the data? -> in planet 'train' folder
        .split_by_idxs(train_idx=train_index, valid_idx=val_index)
        #How to split in train/valid? -> randomly with the default 20% in valid
        .label_from_df(cols='label')
        #How to label? -> use the second column of the csv file and split the tags by ' '
        .transform(tfms, size=224)
        #Data augmentation? -> use tfms with a size of 224
        .databunch(num_workers = 2)).normalize(imagenet_stats)
        #Finally -> use the defaults for conversion to databunch
        
    learner= cnn_learner(data, models.densenet121,ps = 0.25, model_dir='/tmp/models/',
                         metrics = accuracy)
    learner.unfreeze()
    learner.fit_one_cycle(20, slice(3e-4,3e-3))

    loss = learner.validate()
    accuracy_list.append(loss)


# In[ ]:


print(accuracy_list)


# In[ ]:


sum = 0
for i in accuracy_list:
    sum = sum + i[0]


# In[ ]:


sum/10

