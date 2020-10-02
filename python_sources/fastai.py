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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from fastai.vision import *


# In[ ]:


TRAIN_DIR = '../input/train_images/'
TEST_DIR = '../input/test_images/'


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.shape,test_df.shape


# In[ ]:


train_df.head()


# In[ ]:


train_df.category_id.value_counts()


# In[ ]:





# In[ ]:


sample_train_df = pd.concat([train_df.iloc[:1000],train_df.loc[train_df['category_id']==22]])
sample_train_df.shape


# In[ ]:


sample_train_df.category_id.value_counts()


# In[ ]:


# %time train = ImageList.from_df(df=train_df,path=TRAIN_DIR,cols='file_name')
# %time test = ImageList.from_df(df=test_df,path=TEST_DIR,cols='file_name')

get_ipython().run_line_magic('time', "train = ImageList.from_df(df=sample_train_df,path=TRAIN_DIR,cols='file_name')")
get_ipython().run_line_magic('time', "test = ImageList.from_df(df=test_df,path=TEST_DIR,cols='file_name')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "data = (train.split_by_rand_pct(seed=22)\n       .label_from_df(cols='category_id')\n       .add_test(test)\n       .transform(get_transforms(),size=256)\n       .databunch())")


# In[ ]:


data.show_batch()


# In[ ]:


data.batch_size


# In[ ]:


len(data.train_ds), len(data.valid_ds), len(data.test_ds),


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=[FBeta(),accuracy], model_dir='../working/model/',path='../working/tmp/')


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


lr = 6e-3
learn.fit_one_cycle(2, slice(lr))


# In[ ]:


lr = 6e-3
learn.fit_one_cycle(2, slice(lr))


# In[ ]:


learn.save('stage_1_sz256_resnet50')


# In[ ]:




