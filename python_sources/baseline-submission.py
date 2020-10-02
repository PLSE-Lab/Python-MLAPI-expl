#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

X_train = pd.read_csv('/kaggle/input/manomanocentralesupelec/X_train.csv')
y_train = pd.read_csv('/kaggle/input/manomanocentralesupelec/y_train.csv')
X_test = pd.read_csv('/kaggle/input/manomanocentralesupelec/X_test.csv')
X_train.shape, y_train.shape, X_test.shape


# In[ ]:


print('Features: ', X_train.columns)
print('Target: ', y_train.columns)


# ## Dummy algorithm : Average Conversion Rate

# ### Estimate WRMSE

# In[ ]:


from sklearn.metrics import mean_squared_error
X_train['conversion_rate'] = y_train['conversion_rate']
X_train['m_total_vu'] = y_train['m_total_vu']

average_conversion_rate = X_train[X_train.s_date <= '2020-01-15']['conversion_rate'].mean()
print('Average conversion rate', average_conversion_rate)
print('Estimated WRMSE:', mean_squared_error(
    y_true=X_train[X_train.s_date > '2020-01-15']['conversion_rate'],
    y_pred=[average_conversion_rate] * (X_train.s_date > '2020-01-15').sum(),
    sample_weight=X_train[X_train.s_date > '2020-01-15']['m_total_vu']
))


# ## Generate submission

# In[ ]:


y_sub = pd.DataFrame({'id': X_test['id'], 'conversion_rate': average_conversion_rate})
y_sub.head()


# In[ ]:


y_sub.to_csv('/kaggle/working/sample_submission.csv', index=False)

