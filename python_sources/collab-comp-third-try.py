#!/usr/bin/env python
# coding: utf-8

# ## Still think features can be reduced without the ML tools - another try.

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


train = pd.read_csv('../input/learn-together/train.csv')
test = pd.read_csv('../input/learn-together/test.csv')


# In[ ]:


train.columns


# In[ ]:


test.columns


# In[ ]:


train = train.iloc[:, :15].join(train.iloc[:,15:-2]                           .dot(range(1,40)).to_frame('Soil_Type1'))                           .join(train.iloc[:,-1])
train.columns


# In[ ]:


train.shape


# In[ ]:


train = train.iloc[:, :11].join(train.iloc[:,11:-2]                           .dot(range(1,5)).to_frame('Wilderness_Area1'))                           .join(train.iloc[:,-2:])
train.columns


# In[ ]:


X = train.drop(['Cover_Type'], axis = 1)
y = train.Cover_Type


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

forest_model = RandomForestClassifier(n_estimators=10)
forest_model.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import mean_absolute_error

val_predictions = forest_model.predict(X_val)
val_mae = mean_absolute_error(y_val,val_predictions)
val_mae


# In[ ]:


test = test.iloc[:, :15].join(test.iloc[:,15:-1].dot(range(1,40)).to_frame('Soil_Type1')) 
test.columns


# In[ ]:


test.shape


# In[ ]:


test = test.iloc[:, :11].join(test.iloc[:,11:-1]                           .dot(range(1,5)).to_frame('Wilderness_Area1'))                           .join(test.iloc[:,-1])
test.columns


# In[ ]:


test_preds = forest_model.predict(test)
output = pd.DataFrame({'Id': test.Id, 'Cover_Type': test_preds.astype(int)})
output.to_csv('submission.csv', index=False)


# This attempt did slightly better than the previous try. Will cross-validation help let's see.
