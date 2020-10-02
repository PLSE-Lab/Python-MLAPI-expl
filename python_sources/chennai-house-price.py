#!/usr/bin/env python
# coding: utf-8

# Prediction using just multiple linear regression
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


test=pd.read_csv('/kaggle/input/chennai-house-pricing-/test.csv')
train=pd.read_csv('/kaggle/input/chennai-house-pricing-/train.csv')


# In[ ]:


train.head()


# In[ ]:


train.describe(include='all')


# In[ ]:


train.columns


# In[ ]:


train.dtypes


# In[ ]:


train.isnull().sum()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


plt.scatter(train['AREA'] ,train['SALES_PRICE'])


# In[ ]:


plt.scatter(train['MZZONE'] ,train['SALES_PRICE'])


# In[ ]:


plt.scatter(train['REG_FEE'] ,train['SALES_PRICE'])


# In[ ]:



train.dropna(inplace=True)
train.isnull().sum()
train.shape


# In[ ]:


X=train[['INT_SQFT', 'DIST_MAINROAD', 'N_BEDROOM',
       'N_BATHROOM', 'N_ROOM', 'REG_FEE', 'COMMIS']]
Y=train['SALES_PRICE']


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm=LinearRegression()


# In[ ]:


lm.fit(X,Y)


# In[ ]:


test.columns


# In[ ]:


test1=test[['INT_SQFT', 'DIST_MAINROAD', 'N_BEDROOM',
       'N_BATHROOM', 'N_ROOM', 'REG_FEE', 'COMMIS']]


# In[ ]:


prediction=lm.predict(test1)


# In[ ]:


prediction[:10]


# In[ ]:




