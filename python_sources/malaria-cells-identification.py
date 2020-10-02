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


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


path = Path('../input/cell_images/cell_images')
path


# In[6]:


path.ls()


# In[7]:


fnames = get_image_files(path/'Uninfected')
fnames[:5]


# In[8]:


ls_uninfected = np.array(fnames)
ls_uninfected.shape


# In[9]:


open_image(path/'Uninfected/C103P64ThinF_IMG_20150918_164250_cell_28.png')


# In[10]:


fnames = get_image_files(path/'Parasitized')
fnames[:5]


# In[11]:


ls_infected = np.array(fnames)
ls_infected.shape


# In[12]:


open_image(path/'Parasitized/C132P93ThinF_IMG_20151004_152257_cell_149.png')


# In[13]:


open_image(path/'Uninfected/C103P64ThinF_IMG_20150918_164250_cell_28.png').shape


# In[14]:


np.random.seed(42)
tfms = get_transforms(flip_vert=True)
data = ImageDataBunch.from_folder(path, train=".", 
                                  valid_pct=0.20, ds_tfms = tfms, bs=32, 
                                  size=224, num_workers=0).normalize(imagenet_stats)


# In[15]:


print(data.classes)


# In[16]:


print(data.c)


# In[17]:


data.show_batch(9, figsize=(10,10))


# In[18]:


learn = cnn_learner(data, models.resnet50, metrics = accuracy, model_dir="/tmp/model/")
learn.fit_one_cycle(4)


# In[19]:


learn.save('stage-1')


# In[20]:


interp = ClassificationInterpretation.from_learner(learn)
losses, idxs = interp.top_losses()
len(data.valid_ds) == len(losses) == len(idxs)


# In[21]:


interp.plot_top_losses(9, figsize=(12,12))


# In[22]:


interp.plot_confusion_matrix(figsize=(10,10), dpi=60)


# In[23]:


interp.most_confused(min_val=2)


# In[24]:


learn.unfreeze()


# In[25]:


learn.fit_one_cycle(2)


# In[28]:


learn.load('stage-1')
learn.lr_find()


# In[29]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(6, max_lr = slice(1e-6, 1e-4))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(12,12))


# In[ ]:


interp.plot_confusion_matrix(figsize=(10,10), dpi=60)


# In[ ]:


preds,y, loss = learn.get_preds(with_loss=True)
# get accuracy
acc = accuracy(preds, y)
print('The accuracy is {0} %.'.format(acc*100))

