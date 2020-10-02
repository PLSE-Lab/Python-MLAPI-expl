#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xgboost
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import cross_validation, metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score


# In[5]:


train = pd.read_csv("../input/bank.csv")


# In[8]:


train.head()


# In[28]:


train.head(15)


# In[10]:


train.info()


# In[11]:


train.describe()


# In[12]:


train.columns


# In[13]:


train.values


# In[14]:


train.job.value_counts()


# In[16]:


train.marital.value_counts()


# In[17]:


train.default.value_counts()


# In[18]:


train.housing.value_counts()


# In[19]:


train.loan.value_counts()


# In[20]:


train.contact.value_counts()    


# In[22]:


train.month.value_counts() 


# In[23]:


train.day_of_week.value_counts() 


# In[24]:


train.poutcome.value_counts() 


# In[27]:


train.shape


# In[29]:


train.head()


# In[ ]:





# In[ ]:




