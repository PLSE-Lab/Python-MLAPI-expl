#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# I will use XGB regressor which is very popular among Kaggle compititor 

# In[3]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import model_selection, preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# In[4]:


train_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')


# In[5]:


#Conversion using label encoder
clean_df = train_set.copy()

for f in clean_df.columns:
    if clean_df[f].dtype == 'object':
        label = preprocessing.LabelEncoder()
        label.fit(list(clean_df[f].values))
        clean_df[f] = label.transform(list(clean_df[f].values))
clean_df.head()


# In[7]:


train_set.head()


# In[8]:


train_y = clean_df.y.values
train_x = clean_df.drop(["y"],axis=1)
train_x = train_x.drop(["ID"],axis=1)
train_x = train_x.values


# In[9]:


train_x


# In[10]:


#Fitting XGB regressor 
model = xgb.XGBRegressor()
model.fit(train_x,train_y)
print (model)


# In[19]:


#Transforming the testset
id_vals = test_set.ID.values

clean_test = test_set.copy()
for f in clean_test.columns:
    if clean_test[f].dtype == 'object':
        label = preprocessing.LabelEncoder()
        label.fit(list(clean_test[f].values))
        clean_test[f] = label.transform(list(clean_test[f].values))
clean_test.fillna((-999), inplace=True)
test = clean_test.drop(['ID'],axis=1)
x_test = test.values

#testset is ready


# In[25]:


#Predict 
output = model.predict(data=x_test)
final_df = pd.DataFrame()
final_df["ID"] = id_vals
final_df["y"] = output

final_df.to_csv('Output.csv',index=False )
final_df.head()


# In[ ]:




