#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


df_train = pd.read_csv("../input/train.csv")
print(df_train.shape)
df_train.head()


# In[ ]:


df_test = pd.read_csv("../input/test.csv")
print(df_test.shape)
df_test.head()


# In[ ]:


# unique identifier per seat/pilot, per crew, per experiemnt
df_train["pilot_key"] = df_train["crew"].astype(str)+df_train.experiment+df_train["seat"].astype(str)
df_test["pilot_key"] = df_test["crew"].astype(str)+df_test.experiment+df_test["seat"].astype(str)

# another key - maybe what the pilot next to you was doing is also relevant ? 
df_train["crew_exp_key"] = df_train["crew"].astype(str)+df_train.experiment
df_test["crew_exp_key"] = df_test["crew"].astype(str)+df_test.experiment


# In[ ]:


df_train["date"] = pd.to_datetime("01-01-2018")
df_test["date"] = pd.to_datetime("01-01-2018")


# In[ ]:


df_train["date"] = df_train["date"] + pd.to_timedelta(df_train["time"], unit='s')
df_test["date"] = df_test["date"] + pd.to_timedelta(df_test["time"], unit='s')

df_test.tail()


# In[ ]:


print("crews, experiments in train: ", df_train[["crew","experiment"]].nunique())
print("crews, experiments in test: ", df_test[["crew","experiment"]].nunique())


# In[ ]:


print(set(df_train.experiment))
print(set(df_test.experiment))


# In[ ]:





# ## Export for time series

# In[ ]:


df_train.to_csv("aviation_fatal_train_raw.csv.gz",index=False,compression="gzip")
df_test.to_csv("aviation_fatal_test_raw.csv.gz",index=False,compression="gzip")


# In[ ]:




