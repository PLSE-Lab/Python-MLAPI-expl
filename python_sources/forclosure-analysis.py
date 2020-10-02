#!/usr/bin/env python
# coding: utf-8

# In[47]:


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


# In[48]:


import sys
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from tqdm import tqdm
warnings.filterwarnings('ignore')


# **Steps in the problem**
# 
# 1. Explore the dataset and create a dataframe. Use merge conditions where required
# 2. Clean the final dataframe. Fill the null values, identify the categorical columns and identify 
# the columns to be worked upon.
# 3. Make a classification model.
# 4. Improve the model as required.

# **1. Explore the Dataset**

# In[49]:


rf_data = pd.read_excel('../input/RF_Final_Data.xlsx')
print (rf_data.shape)
rf_data.head()


# In[50]:


rf_data.isnull().sum()


# In[51]:


rf_data = rf_data.drop(['Preprocessed_EmailBody','Preprocessed_Subject'], 1)


# In[52]:


cust_data = pd.read_excel('../input/Customers_31JAN2019.xlsx')
print (cust_data.shape)
cust_data.head()


# In[53]:


cust_data.isnull().sum()


# In[54]:


cust_data = cust_data.drop(['PROFESSION','OCCUPATION','POSITION','PRE_JOBYEARS'], 1)


# In[55]:


lms_data = pd.read_excel('../input/LMS_31JAN2019.xlsx')
print (lms_data.shape)
lms_data.head()


# In[56]:


lms_data['CITY'].value_counts()


# In[57]:


lms_data['CITY'].nunique()


# In[58]:


lms_data['PRODUCT'].value_counts()


# In[59]:


lms_data.isnull().sum()


# In[60]:


lms_data = lms_data.drop(['NPA_IN_LAST_MONTH', 'NPA_IN_CURRENT_MONTH', 'CITY'], 1)


# In[61]:


lms_data.AGREEMENTID.nunique()


# In[62]:


cat_columns = ['PRODUCT']
lms_data = pd.get_dummies(columns = cat_columns, data = lms_data)


# In[63]:


useless_columns = ['INTEREST_START_DATE', 'AUTHORIZATIONDATE', 'LAST_RECEIPT_DATE', 'SCHEMEID']
lms_data = lms_data.drop(useless_columns, 1)


# In[64]:


lms_data.head()


# In[65]:


lms_data = lms_data.groupby(['AGREEMENTID']).mean()
lms_data = lms_data.reset_index()


# In[66]:


lms_data.head()


# In[67]:


train_data = pd.read_csv('../input/train_foreclosure.csv')
print (train_data.shape)
train_data.head()


# In[68]:


test_data = pd.read_csv('../input/test_foreclosure.csv')
print (test_data.shape)
test_data.head()


# In[69]:


lms_data.head()


# In[70]:


lms_train_data = pd.merge(train_data, lms_data, on = 'AGREEMENTID')
lms_train_data.shape


# In[71]:


lms_train_data.head()


# In[72]:


lms_train_data.isnull().sum()


# In[73]:


mean_value = lms_train_data['LAST_RECEIPT_AMOUNT'].mean()
lms_train_data['LAST_RECEIPT_AMOUNT'] = lms_train_data['LAST_RECEIPT_AMOUNT'].fillna(value = mean_value)


# In[74]:


lms_train_data = lms_train_data.drop(['CUSTOMERID'], 1)


# In[75]:


lms_test_data = pd.merge(test_data, lms_data, on = 'AGREEMENTID')
lms_test_data.shape


# In[76]:


lms_test_data.head()


# In[77]:


lms_test_data.isnull().sum()


# In[78]:


lms_test_data = lms_test_data.drop(['CUSTOMERID'], 1)
mean_value = lms_test_data['LAST_RECEIPT_AMOUNT'].mean()
lms_test_data['LAST_RECEIPT_AMOUNT'] = lms_test_data['LAST_RECEIPT_AMOUNT'].fillna(value = mean_value)


# In[ ]:





# In[79]:


y = lms_train_data['FORECLOSURE']
X_ID = lms_train_data['AGREEMENTID']
X = lms_train_data.drop(['FORECLOSURE', 'AGREEMENTID'], 1)


# In[80]:


from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb


# In[81]:


xgb.fit(X, y)


# In[82]:


X_test_ID = lms_test_data['AGREEMENTID']
X_test = lms_test_data.drop(['AGREEMENTID', 'FORECLOSURE'], 1)


# In[83]:


xgb_pred = xgb.predict(X_test)


# In[84]:


submission = pd.DataFrame(columns = ['AGREEMENTID', 'FORECLOSURE'])
submission['AGREEMENTID'] = X_test_ID
submission['FORECLOSURE'] = xgb_pred


# In[86]:


submission.to_csv('submission.csv', index = False)


# In[ ]:




