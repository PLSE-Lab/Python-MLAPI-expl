#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd 
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import datetime


# In[ ]:


train = pd.read_csv("../input/train.csv", parse_dates=['date_time'], nrows=10000000)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# tall = train.ix[:, 'date_time']

# In[ ]:


mask = train.is_booking == True


# In[ ]:


trainm = train[mask]


# In[ ]:


#understanding the numerical content 

b1 = trainm[['srch_adults_cnt','srch_children_cnt','srch_rm_cnt']]


# In[ ]:


sns.countplot(y='srch_children_cnt', hue='srch_rm_cnt', data=trainm)


# In[ ]:


# putting the above separately
sns.countplot(y='srch_adults_cnt', hue='srch_rm_cnt', data=trainm)


# In[ ]:


# putting the above separately
sns.countplot(y='srch_adults_cnt', hue='srch_children_cnt', data=trainm)


# In[ ]:


b1.to_csv('rooms_children_adults_nrows1K.csv',header=True)


# In[ ]:


# putting the above separately
sns.countplot(y='srch_adults_cnt', data=trainm)


# In[ ]:


# putting the children
sns.countplot(y='srch_children_cnt', data=trainm)

