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


# In[ ]:


from fastai.tabular import *


# In[ ]:


path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
df.head()


# In[9]:


df.head()


# In[10]:


dep_var = 'salary'
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship','race']
cont_names = ['age','fnlwgt','education-num']
procs = [FillMissing, Categorify, Normalize]


# In[11]:


test = TabularList.from_df(df.iloc[800:1000].copy(), path=path, cat_names=cat_names, cont_names=cont_names)


# In[12]:


data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
       .split_by_idx(list(range(800,1000)))
       .label_from_df(cols=dep_var)
       .add_test(test)
       .databunch())


# In[13]:


data.show_batch()


# In[15]:


learn = tabular_learner(data, layers=[200,100], metrics=accuracy)


# In[16]:


learn.fit(1, 1e-2)


# In[ ]:





# # Inference

# In[17]:


row = df.iloc[0]


# In[18]:


learn.predict(row)


# In[20]:


df['salary'].value_counts()


# In[ ]:




