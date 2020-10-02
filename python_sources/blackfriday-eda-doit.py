#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 
import seaborn as sns

import os
print(os.listdir("../input"))

get_ipython().run_line_magic('matplotlib', 'inline')


# In[43]:


df = pd.read_csv('../input/BlackFriday.csv')
df.sample(5, random_state=1984)


# In[44]:


features = [ 'Gender', 'Age', 'Occupation', 'City_Category',
            'Stay_In_Current_City_Years', 'Marital_Status',
            'Product_Category_1', 'Product_Category_2', 'Product_Category_3']
print("num_features: ", len(features))


# In[45]:


fig, axs_tup = plt.subplots(3, 3, figsize=(20,10),
                            gridspec_kw=dict(hspace=0.5))
for i in range(len(features)):
    df[features[i]].value_counts().plot(kind='bar', title='Value counts of {}'.format(features[i]), ax=axs_tup[int(i/3),i%3])
plt.show()


# In[46]:


fig, axs_tup = plt.subplots(3, 3, figsize=(20,10),
                            gridspec_kw=dict(hspace=0.5))
for i in range(len(features)):
    sns.boxplot(y="Purchase", x= features[i], data=df , ax=axs_tup[int(i/3),i%3])
plt.show()


# In[47]:


# How many null values are there?
df.isnull().sum() / df.isnull().count()


# In[48]:


df['Purchase'].plot(kind='hist', bins=40, alpha=0.7, grid=True, figsize=(14,4))
plt.show()


# ## Feature Extraction

# In[49]:


# We would like to use XGBoost, we aknowledge that as of May 2019 it doesn't support categoricals features natively as LighGBM


# In[59]:


from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split


# In[58]:


label_transformer = OrdinalEncoder()
X_all = df[features].copy()
X_all.loc[:,features] = label_transformer.fit_transform(df[features].fillna(-1))
X_all.head()


# In[56]:


y_all = df['Purchase'].copy()


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.05)


# ## Train a model 

# In[61]:


import xgboost as xgb


# In[66]:


bst = xgb.XGBRegressor(min_child_weight=100, n_estimators=10000)                    .fit(X_train, y_train, verbose=30, eval_set=[(X_test, y_test)], early_stopping_rounds=5)


# In[76]:


df_prediction = pd.DataFrame()
df_prediction['true'] = y_test
df_prediction['prediction'] = bst.predict(X_test)

df_prediction.plot(x='true', y='prediction', kind='scatter', lw=0, s=0.5, figsize=(10,5), c='indigo')
plt.show()


# In[ ]:




