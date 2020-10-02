#!/usr/bin/env python
# coding: utf-8

# ## Dependencies and Global variables

# In[24]:


import numpy as np
import pandas as pd

import torch

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[25]:


# Reload latest version of dependencies
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[26]:


# Fast AI Imports
from fastai import *
from fastai.vision import *


# In[27]:


# Set batchsize
bs = 64


# ## DataBunch Setup

# In[28]:


path = Path('../input')
path_train = path/'train/train'
path_test = path/'test/test/'
path, path_train, path_test


# In[29]:


labels_df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'sample_submission.csv')
labels_df.head()


# In[30]:


np.random.seed(42)
test = ImageList.from_df(test_df, path=path_test)
data = (
    ImageList.from_df(labels_df, path=path_train)
                     .split_by_rand_pct(0.01)
                     .label_from_df()
                     .add_test(test)
                     .transform(get_transforms(
                         flip_vert = True,
                     ), size = 128)
                     .databunch(path=path, bs = bs).normalize(imagenet_stats)
)


# We've used the Fast.ai *DataBlock* API to create a *databunch*. 
# 1. There are some data augmentations that Fast.ai performs by default. These have been tweaked with
# 2. The images are resized to 128x128 instead of 32x32, this led to a higher accuracy on the validation set
# 3. The validation set is set up to be 10% of the training data

# In[31]:


data


# In[32]:


data.show_batch(rows = 3, figsize = (10,8))


# The *1* or *0* indicates above the image indicates whether it has a cactus or not. Our validation set is 20% of our train data

# In[33]:


# Print classes of our classification problem
data.classes


# ## Resnet 101: Training

# In[34]:


learn = cnn_learner(data, models.resnet101, metrics = accuracy, model_dir='/tmp/model/')


# In[35]:


learn.lr_find()


# In[36]:


learn.recorder.plot()


# From the above plot, we will choose the learning rate as the x-axis value corresponding to the steepest descent (without bumps) of the y-axis value, *Loss*.

# In[37]:


lr = 3e-02


# In[38]:


learn.fit_one_cycle(3, slice(lr))


# In[39]:


learn.save('resnet-101-1')


# An near perfect accuracy but this leads to a leaderboard score of 0.9999. Let's try and improve this with *Densenet 161*

# ## DenseNet 161 Training

# In[40]:


learn = cnn_learner(data, models.densenet161, metrics = accuracy, model_dir='/tmp/model/')


# In[41]:


learn.lr_find()


# In[42]:


learn.recorder.plot()


# In[43]:


lr = 3e-02


# In[44]:


learn.fit_one_cycle(3, slice(lr))


# In[45]:


learn.save('densenet-161-1')


# ## Generating Predictions

# ### Resnet 101

# In[46]:


learn = cnn_learner(data, models.resnet101, metrics = accuracy, model_dir='/tmp/model/')


# In[47]:


learn.load('resnet-101-1');


# In[48]:


preds, _ = learn.get_preds(ds_type=DatasetType.Test)


# In[49]:


preds[:, 0]


# In[50]:


test_df['has_cactus'] = np.array(preds[:, 0])
test_df.head()


# In[51]:


test_df.to_csv('submission_resnet_101.csv', index = False)


# ### Densenet 161

# In[52]:


learn = cnn_learner(data, models.densenet161, metrics = accuracy, model_dir='/tmp/model/')


# In[53]:


learn.load('densenet-161-1');


# In[54]:


preds, _ = learn.get_preds(ds_type=DatasetType.Test)


# In[55]:


preds[:, 0]


# In[56]:


test_df['has_cactus'] = np.array(preds[:, 0])
test_df.head()


# In[57]:


test_df.to_csv('submission_densenet_161.csv', index = False)


# In[58]:


from IPython.display import FileLinks
FileLinks('.')


# In[ ]:




