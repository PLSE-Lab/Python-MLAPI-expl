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

from fastai import *
from fastai.tabular import *

import os
# Any results you write to the current directory are saved as output.


# # Load Data

# In[2]:


root=Path("../input")
train_path=root/'train.csv'
test_path=root/'test.csv'


# In[3]:


test_df=pd.read_csv(test_path).fillna(0)
test_df.head()


# In[4]:


train_df=pd.read_csv(train_path)
train_df.head()


# In[5]:


train_df1=train_df.loc[0:1200]
train_df2=train_df.loc[1200:]


# In[6]:


procs=[FillMissing,Categorify]


# In[7]:


valid_idx=list(range(500,700))


# In[8]:


dep_var='SalePrice'

# TODO Feature selection
cat_names=list(train_df.select_dtypes('object').columns.values)
cont_names=list(train_df.select_dtypes(['int64','float64']))
cont_names.remove('Id')
cont_names.remove('SalePrice')


# In[9]:


test_db = TabularList.from_df(test_df,path=test_path,cont_names=cont_names,cat_names=cat_names)


# In[33]:


data=(TabularList.from_df(train_df,path=train_path,cont_names=cont_names,cat_names=cat_names,procs=procs)
#       .split_by_idx(valid_idx)
      .split_by_rand_pct(0.1)
      .label_from_df(cols=dep_var,label_cls=FloatList,log=False)
      .add_test(test_db)
      .databunch())


# In[34]:


data.show_batch(10)


# # Train model

# In[48]:


learn=tabular_learner(data,layers=[4000,2000], metrics=[mean_absolute_error],path='.')


# In[49]:


learn.fit_one_cycle(5,.01)


# In[50]:


learn.save('s1')


# In[51]:


learn.load('s1');


# In[52]:


learn.unfreeze()


# In[53]:


learn.lr_find(1e-10,1e+10)


# In[54]:


learn.recorder.plot()


# In[55]:


learn.fit_one_cycle(20,10)


# # Evaluation

# In[56]:


vdf=train_df2


# In[57]:


vdb=(TabularList.from_df(vdf,path=train_path,cont_names=cont_names,cat_names=cat_names,procs=procs)
     .split_by_rand_pct(.01)
     .label_from_df(cols=dep_var,label_cls=FloatList,log=False)
     .databunch())


# In[58]:


learn.validate(vdb.valid_dl)


# In[59]:


learn.validate(data.valid_dl)


# # Make predictions

# In[24]:


i=2


# In[25]:


data.valid_ds[i]


# In[26]:


train_df1.iloc[500+i]["GarageArea"]


# In[27]:


learn.predict(train_df1.iloc[500+i])[0].obj[0]


# In[32]:


j=10
train_df2.iloc[j]['SalePrice']


# In[29]:


learn.predict(train_df2.iloc[j])


# # Prepare Submissions

# In[30]:


sub_df = pd.DataFrame(columns=['Id','SalePrice']).astype({'Id':int,'SalePrice':float})


# In[31]:


for index, rw in test_df.iterrows():
    price=learn.predict(rw)[0].obj[0]
    sub_df.loc[index]=[int(rw.Id),price]


# In[ ]:


sub_df=sub_df.astype({'Id':int,'SalePrice':float})
sub_df.head()


# In[ ]:


sub_df.to_csv('submission.csv',index=False)

