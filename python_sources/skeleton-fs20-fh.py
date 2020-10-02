#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# # prepare data

# In[ ]:


train_data = pd.read_csv('/kaggle/input/machine-learning-lab-cas-data-science-fs-20/houses_train.csv', index_col=0)


# In[ ]:


X_train = train_data.drop(columns='price')
y_train = train_data['price']


# In[ ]:


X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, stratify=X_train['object_type_name'], test_size=0.1)


# # define and train model

# In[ ]:





# # Predict and evaluate prices for dev set

# In[ ]:


y_dev_pred = np.random.randint(10000, 2000000, X_dev.shape[0])


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[ ]:


mean_absolute_percentage_error(y_dev, y_dev_pred)


# # Predict prices for test set

# In[ ]:


X_test = pd.read_csv('/kaggle/input/machine-learning-lab-cas-data-science-fs-20/houses_test.csv', index_col=0)


# 

# In[ ]:


y_test_pred = np.random.randint(10000, 2000000, X_test.shape[0])


# In[ ]:


X_test_submission = pd.DataFrame(index=X_test.index)


# In[ ]:


X_test_submission['price'] = y_test_pred


# In[ ]:


X_test_submission.to_csv('random_submission.csv', header=True, index_label='id')


# In[ ]:





# In[ ]:




