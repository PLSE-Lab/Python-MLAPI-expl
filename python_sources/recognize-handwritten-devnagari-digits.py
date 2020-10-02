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
print(os.listdir("../input/devanagari-character-set/Images/Images"))

# Any results you write to the current directory are saved as output.tfms = get_transforms(do_flip=False)


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


PATH = "../input/devanagari-character-set/Images/Images"


# In[ ]:


np.random.seed(42)
tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(PATH, valid_pct=0.2, ds_tfms=tfms, size=32, num_workers=0)


# In[ ]:


data.classes


# In[ ]:


data.show_batch(rows=3, figsize=(5,5))


# In[ ]:


cache_dir = os.path.expanduser(os.path.join('~','.torch'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)


# In[ ]:


models_dir = os.path.join(cache_dir,'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


# In[ ]:


get_ipython().system('cp ../input/resnet34/resnet34.pth ~/.torch/models/resnet34-333f7ec4.pth ')


# In[ ]:


MODEL_PATH = '/tmp/models'
learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir=MODEL_PATH)


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1')
learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4,max_lr=slice(1e-05,1e-04))


# In[ ]:


learn.save('stage-2')


# In[ ]:




