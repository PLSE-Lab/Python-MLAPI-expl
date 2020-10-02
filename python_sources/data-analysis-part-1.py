#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv("/kaggle/input/prostate-cancer-grade-assessment/train.csv")


# In[ ]:


train.head()


# In[ ]:


test = pd.read_csv("/kaggle/input/prostate-cancer-grade-assessment/test.csv")


# In[ ]:


test.head()


# In[ ]:


train.shape[0]


# In[ ]:


test.shape[0]


# In[ ]:


train.info()


# In[ ]:


train["isup_grade"].value_counts().plot(kind='bar')


# In[ ]:


train["gleason_score"].value_counts().plot(kind='bar')


# In[ ]:


provider_isup = train.groupby("data_provider").sum()


# In[ ]:


provider_isup.plot(kind='bar')


# In[ ]:


data_provider_gleason_score = train.groupby("data_provider")["gleason_score"].value_counts()


# In[ ]:


plt.figure(figsize=(10,6))
data_provider_gleason_score.plot(kind="bar")


# In[ ]:




