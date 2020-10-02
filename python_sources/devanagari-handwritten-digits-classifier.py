#!/usr/bin/env python
# coding: utf-8

# # Devanagari Handwritten Digits Classifier - Fast.ai (Lesson 1)

# ## Loading Libraries

# In[1]:


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


# In[2]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from fastai import *
from fastai.vision import *


# In[4]:


import numpy as np
import pandas as pd


# ## Defining the path for data

# In[5]:


path = Path('../input/devanagari')
path.ls()


# In[6]:


fnames = get_image_files(path/'test/23')
fnames[:5]


# In[7]:


open_image(path/'test/23/31991.png').shape


# In[8]:


tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, train='train', valid='test', size=32, bs=32).normalize(imagenet_stats)


# In[9]:


print(data.classes)
print(len(data.classes))
print(data.c)


# In[10]:


learn = cnn_learner(data, models.resnet50, metrics=accuracy, model_dir="/tmp/model/")
learn.fit_one_cycle(4)


# In[11]:


data.show_batch(3, figsize=(5,5))


# In[12]:


learn.save('stage-1')


# In[13]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[14]:


interp.plot_top_losses(9, figsize=(10,10))


# In[15]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[16]:


interp.most_confused(min_val=2)


# In[17]:


learn.unfreeze()


# In[18]:


learn.fit_one_cycle(3)


# In[19]:


learn.load('stage-1');
learn.lr_find()


# In[20]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(8, max_lr=slice(1e-6, 1e-4))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(12,12))


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


preds,y, loss = learn.get_preds(with_loss=True)
# get accuracy
acc = accuracy(preds, y)
print('The accuracy is {0} %.'.format(acc*100))


# In[ ]:




