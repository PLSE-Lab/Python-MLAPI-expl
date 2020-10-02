#!/usr/bin/env python
# coding: utf-8

# # Hi!
# 
# This is Amitava of team AMITAVA_Nidhi_Leur_PRISHAT_Hristo. I took the privilege to change the leader to Hristo noticing that he has been the one that is most active so far. I can be reached at @avamaity till we decide on a common channel for colaboration.

# ## First attempt - just applying the basics

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/learn-together/train.csv")
test = pd.read_csv("../input/learn-together/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.tail()


# In[ ]:


train.isnull().sum().sum()


# In[ ]:


test.isnull().sum().sum()


# In[ ]:


X = train.drop(['Cover_Type'], axis = 1)
y = train.Cover_Type


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


# In[ ]:


forest_model = DecisionTreeRegressor(random_state=0)
forest_model.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import mean_absolute_error

val_predictions = forest_model.predict(X_val)
val_mae = mean_absolute_error(y_val,val_predictions)
val_mae


# In[ ]:


test_preds = forest_model.predict(test)
output = pd.DataFrame({'Id': test.Id, 'Cover_Type': test_preds.astype(int)})
output.to_csv('submission.csv', index=False)
