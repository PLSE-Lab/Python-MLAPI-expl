#!/usr/bin/env python
# coding: utf-8

# In[101]:


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


# In[102]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import os
from fastai.tabular import *


# In[164]:


df = pd.read_csv('../input/train.csv').drop('ID_code', axis=1)
test_df = pd.read_csv('../input/test.csv')
valid_idx = random.sample(list(df.index.values), int(len(df)*0.05))


# In[177]:


features = [feature for feature in df.columns if 'var_' in feature]
features


# In[166]:


data = TabularDataBunch.from_df(path='.', df=df, dep_var='target', valid_idx=valid_idx,
                                    cat_names=[], cont_names=features, procs=[FillMissing, Normalize], test_df=test_df)


# In[167]:


learn = tabular_learner(data, layers=[200, 100, 50], metrics=[accuracy])


# In[169]:


learn.lr_find(end_lr=1e3)
learn.recorder.plot()


# In[170]:


learn.fit_one_cycle(1, max_lr=5e-2)
learn.recorder.plot_losses()


# In[171]:


test_pred, test_y = learn.get_preds(ds_type=DatasetType.Test)


# In[178]:


print(test_y[0:20])
test_y.sum()


# In[180]:


valid_pred, valid_y = learn.get_preds(ds_type=DatasetType.Valid)
print(valid_y[0:20])
valid_y.sum()


# In[ ]:




