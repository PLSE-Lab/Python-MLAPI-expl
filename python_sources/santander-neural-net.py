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


# In[3]:


from fastai.tabular import *


# In[4]:


path = Path('../input')


# In[5]:


get_ipython().system('ls {path}')


# In[6]:


df = pd.read_csv(f'{path}/train.csv', index_col='ID_code')


# In[7]:


df.head()


# In[8]:


df.describe()


# In[9]:


# Describe the df where the target == 1 
df[df['target']==1].describe()


# In[10]:


len(df)


# In[11]:


# describe the df where target == 0
df[df['target']==0].describe()


# In[12]:


df_test = pd.read_csv(f'{path}/test.csv', index_col='ID_code')


# In[13]:


df_test.describe()


# In[14]:


df['var_0'].unique()


# In[15]:


for n in df.columns:
    print(n, ':', len(df[n].unique()))


# In[16]:


dep_var = 'target'


# In[17]:


cont_list, cat_list = cont_cat_split(df=df, max_card=20, dep_var=dep_var)


# In[18]:


procs = [FillMissing, Categorify, Normalize]


# In[19]:


test = TabularList.from_df(df_test, cat_names=cat_list, cont_names=cont_list, procs=procs)


# In[43]:


data = (TabularList.from_df(df, path=path, cont_names=cont_list, cat_names=cat_list, procs=procs)
        .split_by_rand_pct(0.1)
        .label_from_df(dep_var)
        .add_test(test, label=0)
        .databunch())


# In[44]:


data.batch_size = 128


# In[45]:


learn = tabular_learner(data, layers=[ 100 , 100], metrics=accuracy, path=('.'), wd=1e-2)


# In[46]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# In[47]:


data.device = device


# In[48]:


learn.lr_find()
learn.recorder.plot()


# In[49]:


learn.fit_one_cycle(6, 5e-03)


# In[ ]:


predictions, *_ = learn.get_preds(DatasetType.Test)


# In[ ]:


predictions


# In[ ]:


labels = predictions[:,1]


# In[ ]:


#learn.fit_one_cycle(3, 1e-02)


# In[ ]:


#labels = np.argmax(predictions, 1)


# In[ ]:


#labels


# In[ ]:


#df_test.index


# In[ ]:


final_df = pd.DataFrame({'ID_code': df_test.index, 'target': labels})


# In[ ]:


final_df.to_csv('submission.csv', index=False)

