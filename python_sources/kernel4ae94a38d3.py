#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# In[ ]:


train_data


# In[ ]:


print ("Train Data Shape ", train_data.shape)
train_data.describe(include='all')


# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data


# In[ ]:


print ("Test Data Shape ", test_data.shape)
test_data.describe(include='all')


# In[ ]:


test_data.isnull().sum()


# In[ ]:


sns.jointplot(x = "popularity",y="revenue",data=train_data)
plt.show()


# In[ ]:


sns.jointplot(x = "runtime",y="revenue",data=train_data)
plt.show()


# In[ ]:


sns.distplot(train_data['revenue'])


# In[ ]:


train_data['log_revenue'] = np.log(train_data['revenue'])
sns.distplot(train_data['log_revenue'])


# In[ ]:


train_data['release_day','release_month','release_year'] = train_data['release_date'].apply(lambda x : x.split('/'))
train_data['release_date'] = pd.to_datetime(train_data['release_date'])
train_data['dayofweek']=train_data['release_date'].dt.dayofweek


# In[ ]:


train_data


# In[ ]:


##leave only important columsn for revenue
## add revenue column in test data as nan
##

