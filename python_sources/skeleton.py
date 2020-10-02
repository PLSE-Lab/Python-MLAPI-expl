#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# # prepare data

# In[12]:


train_data = pd.read_csv('../input/houses_train.csv', index_col=0)


# In[13]:


X_train = train_data.drop(columns='price')
y_train = train_data['price']


# In[14]:


X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, stratify=X_train['object_type_name'], test_size=0.1)


# # define and train model

# In[ ]:





# # Predict and evaluate prices for dev set

# In[15]:


y_dev_pred = np.random.randint(10000, 2000000, X_dev.shape[0])


# In[16]:


def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[17]:


mean_absolute_percentage_error(y_dev, y_dev_pred)


# # Predict prices for test set

# In[18]:


X_test = pd.read_csv('../input/houses_test.csv', index_col=0)


# 

# In[19]:


y_test_pred = np.random.randint(10000, 2000000, X_test.shape[0])


# In[20]:


X_test_submission = pd.DataFrame(index=X_test.index)


# In[21]:


X_test_submission['price'] = y_test_pred


# In[22]:


X_test_submission.to_csv('random_submission.csv', header=True, index_label='id')


# In[ ]:





# In[ ]:




