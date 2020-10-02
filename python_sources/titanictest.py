#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC


# In[2]:


# Load in the train and test datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[3]:


train.head(5)


# In[4]:


test.head(5)


# In[5]:


train.info()


# In[6]:


train.describe()


# In[7]:


pd.isna(train).describe()


# In[8]:


train.iloc[:,[0, 1, 2, 5, 6, 7, 9]].apply(lambda x:x.mean())


# In[9]:


train.Pclass.value_counts()


# In[10]:


len(train.Cabin.unique())


# In[11]:


train.Age.mean()


# In[12]:


train.Age.unique()


# In[13]:


train.groupby('Embarked').size()


# In[ ]:




