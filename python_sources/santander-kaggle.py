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


from fastai import *
from fastai.tabular import *


# In[ ]:


df_train = pd.read_csv("../input/train.csv")
#df_train.head()


# In[ ]:


df_test = pd.read_csv("../input/test.csv")
#df_test.head()


# In[ ]:


df_sample_submission = pd.read_csv("../input/sample_submission.csv")
df_sample_submission.head()


# In[ ]:


#df_train.describe()


# In[ ]:


#df_train['target'].value_counts()


# In[ ]:


#I have 202 columns Total
#df_train.columns


# In[ ]:


#cont_names = train.iloc[:,1:].columns.tolist()
cont_names = df_train.iloc[:,2:].columns.tolist()
#cont_names


# In[ ]:


dep_var = 'target'
procs = [FillMissing, Categorify, Normalize]
#valid_idx = range(len(df_train)-20000, len(df_train))
#.split_by_idx(valid_idx)
data = (TabularList.from_df(df_train, procs=procs, cont_names=cont_names)
        .split_by_rand_pct(0.10)
        .label_from_df(cols=dep_var)
        .add_test(TabularList.from_df(df_test,cont_names=cont_names, procs=procs))
        .databunch())


# In[ ]:


#learn = tabular_learner(data, layers=[1000,500],emb_drop=0.3, metrics=accuracy)
learn = tabular_learner(data, layers=[200,100], metrics=accuracy)


# In[ ]:


#learn.model


# In[ ]:


#data.show_batch(rows=4)


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion = True)


# In[ ]:


learn.fit_one_cycle(2, 5e-01,wd=0.2)


# In[ ]:


learn.save('1')


# In[ ]:


learn.load('1');


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion = True)


# In[ ]:


learn.fit_one_cycle(3, 1e-2)


# In[ ]:


predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions, 1)


# In[ ]:


sub_df = pd.DataFrame({'ID_code': df_test['ID_code'], 'target': labels})
sub_df.to_csv('submission.csv', index=False)


# In[ ]:


sub_df.tail()

