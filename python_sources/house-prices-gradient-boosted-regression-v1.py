#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Any results you write to the current directory are saved as output.


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

print(df_train.shape)
print(df_test.shape)


# In[ ]:


df_train.info()


# In[ ]:


df_null = pd.DataFrame(df_train.isnull().sum().tolist(),columns=['adet'])
df_null['columns'] = df_train.columns.tolist()
df_null['perc'] =(df_null.adet / df_train.shape[0] * 100).map("{:,.2f}".format)
df_null.sort_values(by='adet', ascending=False)[df_null.adet > 0]


# There are so many null values in some columns like PoolQC, MiscFeature etc.
# We can try fill this columns or just drop. And I select first 5 column in df_null for dropping. Others will be filled with mean. For categorical data we need to use like one hot encoder. But we have to dataset train and set, so one hot encoder converts  all categories to column. If we use one hot encode (or any encoder), we have to concat two dataset together before encoding, and fit after that. 

# In[ ]:


df_train.drop(labels=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'], axis=1, inplace=True)

for col in df_train.columns:
    if df_train[col].dtypes != 'object':
        if df_train[col].isnull().sum()>0:
            df_train[col].fillna(value=df_train[col].mean(), inplace=True)    
#do the same for df_test
df_test.drop(labels=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'], axis=1, inplace=True)

for col in df_test.columns:
    if df_test[col].dtypes != 'object':
        if df_test[col].isnull().sum()>0:
            df_test[col].fillna(value=df_test[col].mean(), inplace=True)   




# In[ ]:


from sklearn.preprocessing import OneHotEncoder

oht = OneHotEncoder()

#check column types and if type=='object' get that. 
obj_col_list=[]
for col in df_train.columns:
    if df_train[col].dtypes == 'object':
        obj_col_list.append(col)
        
df_ = pd.DataFrame(columns=obj_col_list)        

df_ = df_train[obj_col_list]
df_ = df_.append(df_test[obj_col_list], ignore_index=True)

df_ = pd.get_dummies(df_) #one hot encoder for all categorical columns.

#after one hot encoder. we can write values to original dataframe

for col in obj_col_list:
    df_train.drop(col, axis=1, inplace=True)
    df_test.drop(col, axis=1, inplace=True)
    
for col in df_.columns:
    df_train[col] = df_[col].iloc[:df_train.shape[0]].values
    df_test[col] = df_[col].iloc[-df_test.shape[0]:].values
       
    
print(df_train.shape)
print(df_test.shape)
print(df_.shape)


# In[ ]:


#drop Id columns in two dataframe

#save ids for submission file
train_ids = df_train['Id'].values
test_ids = df_test['Id'].values 

df_train.drop('Id', axis=1, inplace=True)
df_test.drop('Id', axis=1, inplace=True)

y = df_train.SalePrice
x = df_train.drop('SalePrice', axis = 1)
print(y.shape)
print(x.shape)


# In[ ]:


#split train data to test and train
train_x = (x.values)[:int(x.shape[0] * 0.80),:]
test_x = (x.values)[int(x.shape[0] * 0.80):,:]

train_y = (y.values)[:int(x.shape[0] * 0.80)]
test_y = (y.values)[int(x.shape[0] * 0.80):]

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)


# In[ ]:


#create model
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(max_depth=10, n_estimators=1000, verbose=1,max_features='sqrt')

model = gbr.fit(train_x, train_y)

print("%.2f" % (model.score(test_x, test_y)*100) )


# In[ ]:


print(df_test.isnull().sum())
#df_test.fillna(value=0, axis=0, inplace=True)
#print(df_test.isnull().sum())


# In[ ]:


preds = model.predict(df_test.values)


# In[ ]:


preds[:10]


# In[ ]:


example_submission = pd.read_csv('../input/sample_submission.csv')
example_submission.head()


# In[ ]:


submission = pd.DataFrame(index=range(df_test.shape[0]), columns=['Id','SalePrice'])
submission.Id = test_ids
submission.SalePrice = preds

submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




