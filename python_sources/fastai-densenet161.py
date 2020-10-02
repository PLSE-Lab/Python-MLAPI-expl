#!/usr/bin/env python
# coding: utf-8

# In[58]:


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


# In[59]:


import os
import numpy as np
from fastai import *
from fastai.vision import *


# In[60]:


train_path = Path('../input/train')
test_path = Path('../input/test')
print(train_path, test_path)


# In[61]:


import pandas as pd

train_df = pd.read_csv('../input/train.csv')
train_df.head(10)


# In[62]:


test_df = pd.read_csv('../input/sample_submission.csv')
# test_df.head(10)
# print(len(test_df), len(train_df))


# In[63]:


test_data = ImageList.from_df(test_df, path = test_path, folder = 'test')

train_data = (ImageList.from_df(train_df, path = train_path, folder = 'train')
            .split_by_rand_pct(0.1)
            .label_from_df()
            .add_test(test_data)
            .transform(get_transforms(), size = 32)
            .databunch(path = '.', bs = 64)
            .normalize(imagenet_stats))


# In[64]:


learner = cnn_learner(train_data, models.densenet161, metrics = [error_rate, accuracy])


# In[65]:


learner.fit_one_cycle(8)


# In[66]:


learner.unfreeze()
learner.fit_one_cycle(4)


# In[67]:


preds, _ = learner.get_preds(ds_type = DatasetType.Test)
test_df.has_cactus = preds.numpy()[:, 0]


# In[68]:


test_df.to_csv('submission.csv', index = False)


# In[ ]:





# In[ ]:




