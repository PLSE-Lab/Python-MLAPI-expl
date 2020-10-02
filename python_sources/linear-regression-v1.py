#!/usr/bin/env python
# coding: utf-8

# In[33]:


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

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.impute import SimpleImputer


# In[2]:


fetch_from = '../input/train.csv'
train = pd.read_csv(fetch_from)


# In[3]:


fetch_from = '../input/test.csv'
test = pd.read_csv(fetch_from)


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


train.describe()


# In[7]:


test.describe()


# In[8]:


train.sample(5)


# In[9]:


train.hist(bins=50, figsize=(20,15))
plt.tight_layout(pad=0.4)
plt.show()


# In[10]:


price = "salesPrice"


# In[11]:


corr_matrix = train.corr()
plt.subplots(figsize=(15,10))
sns.heatmap(corr_matrix, vmax=1.0, square=True, cmap="Blues")


# In[12]:


corr_matrix = train.corr()
corr_matrix["SalePrice"].sort_values(ascending=False)[:20]


# In[13]:


train.isnull().sum().sum()


# In[14]:


test.isnull().sum().sum()


# In[15]:


test.isnull().sum()


# In[ ]:





# In[16]:


train_fe = train.copy()
test_fe = test.copy()


# In[17]:


train_ID = train_fe['Id']
test_ID = test_fe['Id']

train_fe.drop(['Id'], axis=1, inplace=True)
test_fe.drop(['Id'], axis=1, inplace=True)


# In[18]:


train_fe.head()


# In[35]:


test_fe.head()


# In[20]:


y = train_fe["SalePrice"]


# In[21]:


X = train_fe[["OverallQual", 'GrLivArea','GarageCars','GarageArea',"TotalBsmtSF","1stFlrSF",
              "FullBath","TotRmsAbvGrd","YearBuilt","YearRemodAdd"]]
X_pred = test_fe[["OverallQual", 'GrLivArea','GarageCars','GarageArea',"TotalBsmtSF","1stFlrSF",
              "FullBath","TotRmsAbvGrd","YearBuilt","YearRemodAdd"]]


# In[22]:


X.head()


# In[54]:


X_pred.head()


# In[57]:


X_pred.isnull().sum()


# In[53]:


X_pred.isnull().index


# Replacing missing values with means in test data set:

# In[68]:


X_pred_nomissing = X_pred.select_dtypes(include=[np.number]).interpolate().dropna()


# In[69]:


X_pred_nomissing.isnull().sum()


# In[73]:


X_pred_nomissing.shape


# In[74]:


X_pred_nomissing.head()


# In[75]:


lm = LinearRegression()


# In[76]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[77]:


lm.fit(X_train,y_train)


# In[78]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[79]:


print(lm.intercept_)


# In[80]:


predictions = lm.predict(X_test)


# In[81]:


plt.scatter(y_test,predictions)


# In[82]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[85]:


X_pred_results = lm.predict(X_pred_nomissing)


# In[88]:


plt.hist(X_pred_results)


# In[90]:


test.Id.shape


# In[91]:


output=pd.DataFrame({'Id':test.Id, 'SalePrice':X_pred_results})


# In[93]:


output.to_csv("submissions.csv", index=False)


# In[ ]:




