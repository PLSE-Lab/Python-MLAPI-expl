#!/usr/bin/env python
# coding: utf-8

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
import csv
bs = 64


# In[4]:


path = untar_data(URLs.PLANET_SAMPLE); path


# In[5]:


path.ls()


# In[6]:


path_img = path/'train'
path_labels = path/'labels.csv'


# In[23]:


fnames = get_image_files(path_img)
fnames[:5]


# In[24]:


csvreader = csv.reader(open(path_labels))
for i,row in enumerate(csvreader):
    print(row)
    if(i >= 5):
        break


# In[30]:


np.random.seed(12)
data = ImageDataBunch.from_csv(path=path, folder='train', ds_tfms=get_transforms(), suffix='.jpg', label_delim=' ', size=224, bs=bs, num_workers=0).normalize(imagenet_stats)


# In[31]:


data.show_batch(rows=3, figsize=(10,10))


# In[32]:


print(data.classes)
len(data.classes),data.c


# In[67]:


learn = cnn_learner(data, models.resnet34, metrics=fbeta)


# In[68]:


learn.fit_one_cycle(4)


# In[69]:


learn.save('stage-1')


# In[96]:


lrf = learn.lr_find()
learn.recorder.plot()


# In[97]:


learn.unfreeze()
learn.fit_one_cycle(4, max_lr=slice(1e-3,1e-2))


# In[98]:


interp = ClassificationInterpretation.from_learner(learn)


# In[100]:


learn.save('stage-2')


# In[101]:


lrf = learn.lr_find()
learn.recorder.plot()


# In[102]:


learn.unfreeze()
learn.fit_one_cycle(4, max_lr=slice(1e-6))


# In[104]:


get_ipython().run_line_magic('pinfo2', 'URLs')


# In[ ]:




