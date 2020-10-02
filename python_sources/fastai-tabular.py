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
from fastai.tabular import *
import os
print(os.listdir("../input"))
path = Path('../input')
x_train = pd.read_csv(path/'train.csv')
# Any results you write to the current directory are saved as output.


# Print the x_train

# In[ ]:


x_train.head()


# Checking for the categorical columns

# In[ ]:


k = x_train['wheezy-copper-turtle-magic']
cat_names = ['wheezy-copper-turtle-magic']


# In[ ]:


procs = [FillMissing,Categorify,Normalize]


# Splitting the DATA

# First getting the size of the Dataframe to get the validating split

# In[ ]:


x_train.shape   #262144 rows and 258 columns
cont_names = x_train.columns.tolist()[1:-1]
cont_names.remove('wheezy-copper-turtle-magic')

cat_names = ['wheezy-copper-turtle-magic']

procs = [FillMissing, Categorify, Normalize]


# In[ ]:


valid_idx = range(len(x_train)- 20000, len(x_train))
x_test = pd.read_csv(path/'test.csv')
dep_var = 'target'

data = TabularDataBunch.from_df(path, x_train, dep_var=dep_var, valid_idx=valid_idx, procs=procs,
                                cat_names=cat_names, cont_names=cont_names, test_df=x_test, bs=2048)


# In[ ]:


learn = tabular_learner(data,layers = [1000, 750, 500, 300],emb_szs={'wheezy-copper-turtle-magic': 200}, metrics=accuracy, ps=0.65, wd=3e-1,model_dir="/tmp/model/")


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr = 1e-2
learn.fit_one_cycle(50, lr)


# In[ ]:


test_pred = learn.get_preds(DatasetType.Test)


# In[ ]:


sub_df = pd.read_csv(path/'sample_submission.csv')
sub_df.target = test_pred[0][:,1].numpy()
sub_df.head()


# In[ ]:


sub_df.to_csv('solution.csv', index=False)


# In[ ]:




