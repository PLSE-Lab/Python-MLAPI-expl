#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')


# ### Goal of this notebook is to do some basic EDA of Elo-Recommendations Data and start developing basic models, step by step.  
# 
# **This notebook will be in progress at least till 12/15/2018**. So please excuse the typos and other rough edges. 

# ### Let us first read the input data

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_merchants = pd.read_csv('../input/merchants.csv')
df_hist_trans = pd.read_csv('../input/historical_transactions.csv')
df_new_merchant_trans = pd.read_csv('../input/new_merchant_transactions.csv')


# ### Display the data frames with titles. 

# In[ ]:


print("Training Data Sample");display(df_train.head())
print("Test Data Sample");display(df_test.head())
print("Merchant Data Sample");display(df_merchants.head())
print("Historical Transactions Sample");display(df_hist_trans.head())
print("New Merchant Transactions Sample");display(df_new_merchant_trans.head())


# ### Let us first understand the training data & test data

# In[ ]:


df_test.dtypes


# It looks like the first_active_month is stored as a string type. Let us change it to datetime format

# In[ ]:


df_train['first_active_month'] = pd.to_datetime(df_train['first_active_month'])
df_test['first_active_month'] = pd.to_datetime(df_test['first_active_month'])


# First, let us look at the training and test data distributions to check if they are similar. 

# In[ ]:


f, axes = plt.subplots(1, 2, figsize=(12,6))
sns.distplot( df_train.feature_1,ax=axes[0], kde = False, color = 'green', bins=10).set_title("Train Data")

sns.distplot( df_test.feature_1,ax=axes[1], kde = False, color = 'red', bins=10).set_title("Test Data") 
axes[0].set(ylabel='Card Counts')
f.suptitle('feture_1 Distributions: Training Data and Test Data')
axes[0].set_xticks(np.arange(1,6,1))
axes[1].set_xticks(np.arange(1,6,1))
plt.show()


# In[ ]:


f, axes = plt.subplots(1, 2, figsize=(12,6))
sns.distplot( df_train.feature_2,ax=axes[0], kde = False, color = 'green', bins=10).set_title("Train Data")

sns.distplot( df_test.feature_2,ax=axes[1], kde = False, color = 'red', bins=10).set_title("Test Data") 
axes[0].set(ylabel='Card Counts')
f.suptitle('feture_2 Distributions: Training Data and Test Data')
axes[0].set_xticks(np.arange(1,4,1))
axes[1].set_xticks(np.arange(1,4,1))
plt.show()


# In[ ]:


f, axes = plt.subplots(1, 2, figsize=(12,6))
sns.distplot( df_train.feature_3,ax=axes[0], kde = False, color = 'green', bins=10).set_title("Train Data")

sns.distplot( df_test.feature_3,ax=axes[1], kde = False, color = 'red', bins=10).set_title("Test Data") 
axes[0].set(ylabel='Card Counts')
f.suptitle('feture_3 Distributions: Training Data and Test Data')
axes[0].set_xticks(np.arange(1,2,1))
axes[1].set_xticks(np.arange(1,2,1))
plt.show()


# In[ ]:


f, axes = plt.subplots(1, 2, figsize=(12,6))
df_test['first_active_month'] = df_test['first_active_month'] .fillna(axis=0, method='ffill')
sns.distplot( df_train.first_active_month,ax=axes[0], kde = False, color = 'green', bins=20).set_title("Train Data")
sns.distplot( df_test.first_active_month,ax=axes[1], kde = False, color = 'red', bins=20).set_title("Test Data") 
axes[0].set(ylabel='Card Counts')
f.suptitle('first_active_month Distributions: Training Data and Test Data')

plt.show()


# ### It can be seen that feature_1, feature_2, feature_3 and first_active month have similar distributions in the train and test data.
# 
# ### Now, let us use a the seaborn pairplot to quickly check if any of the input variables in the training data are clearly correlated with the target variable. 

# In[ ]:


plt.figure(figsize=(12,8))
sns.pairplot(df_train.loc[:,df_train.columns != 'card_id'])


# From the above figure, it is not immediately clear if any of the feature_1, feature_2, feature_3 are correlated with the target variable. 
# 
# ### Let us start with a linear model
# 
# Since time stamps are more dificult to work with for linear models, let us start convert the first_active_moth feature, into a customer age feaure, which just returns the number of days since the customer first registered. 

# In[ ]:


latest_date = max(df_train['first_active_month'])
df_train['customer_age'] = (latest_date - df_train['first_active_month']).astype('timedelta64[D]')
df_test['customer_age'] = (latest_date - df_test['first_active_month']).astype('timedelta64[D]')
sns.pairplot(df_train.loc[:,df_train.columns != 'card_id'])


# **From the above, it looks like the customer age feature ma have some correlation with the target variable after all. **
# 
# **Let us generate a linear model and submit it, as a first step. **

# In[ ]:


# split the training set into training set and validation set 
# We are not going to do K-fold cross validation for now. We will do that in a later model

X_train, X_val, y_train, y_val = train_test_split(df_train[['feature_1','feature_2','feature_3','customer_age']],df_train['target'], test_size=0.1, random_state=42)
#X_train.shape, X_val.shape, y_train.shape, y_val.shape
linear_model = LinearRegression().fit(X_train,y_train)
y_pred = linear_model.predict(X_val)
print("Root Mean squared error: %.2f" % np.sqrt(mean_squared_error(y_val, y_pred)))


# ## Submit the best model so far

# In[ ]:


y_test = linear_model.predict(df_test[['feature_1', 'feature_2', 'feature_3','customer_age']])
linear_submission = pd.DataFrame({"card_id": df_test["card_id"].values})
linear_submission['target'] = y_test
linear_submission.to_csv('linear_submission.csv',index=False)


# <font color='red'>In progress. More to come.. </font>
