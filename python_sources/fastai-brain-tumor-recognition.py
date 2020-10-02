#!/usr/bin/env python
# coding: utf-8

# ## Please upvote if you like it ;)

# # Import Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pathlib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import gc
print(os.listdir("../input"))

from sklearn.metrics import confusion_matrix
from fastai import *
from fastai.vision import *

# Any results you write to the current directory are saved as output.


# # Make Data

# In[ ]:


DATA_DIR='../input/brain_tumor_dataset'


# In[ ]:


os.listdir(f'{DATA_DIR}')


# In[ ]:


data = ImageDataBunch.from_folder(DATA_DIR, train=".", 
                                  valid_pct=0.2,
                                  ds_tfms=get_transforms(flip_vert=True, max_warp=0),
                                  size=224,bs=10, 
                                  num_workers=0).normalize(imagenet_stats)


# In[ ]:


print(f'Classes: \n {data.classes}')


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# # Model Build

# In[ ]:


learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/")


# # Train

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(6,1e-2)


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(6,1e-6)


# In[ ]:


learn.save('stage-2')


# # Check Result

# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(8,8), dpi=60)


# ## Please upvote if you like it ;)
