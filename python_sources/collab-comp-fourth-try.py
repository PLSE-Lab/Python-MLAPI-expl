#!/usr/bin/env python
# coding: utf-8

# # Persisting with basics

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


#reading the files
train = pd.read_csv("../input/learn-together/train.csv")
test = pd.read_csv("../input/learn-together/test.csv")


# In[ ]:


#some basic data characteristics
a = (train[train.columns[15:-1]]==1).sum() 
b = (test[test.columns[15:]]==1).sum() 
print(pd.concat([a.rename('train'),b.rename('test')], axis=1))


# In[ ]:


#Seems like cols Soil_Type7 and Soil_Type15 can be dropped without much affecting accuracy
c = (train[train.columns[11:15]]==1).sum() 
d = (test[test.columns[11:15]]==1).sum() 
print(pd.concat([c.rename('train'),d.rename('test')], axis=1))


# In[ ]:


#The distribution of Wilderness Area appears to be ok.
#dropping Soil_Type7 and Soil_Type15
train = train.drop(['Id','Soil_Type7', 'Soil_Type15'], axis = 1)
testids = test['Id']
test = test.drop(['Id','Soil_Type7', 'Soil_Type15'], axis = 1)


# In[ ]:


#preparing data for training the model
X = train.drop(['Cover_Type'], axis = 1)
y = train.Cover_Type


# In[ ]:


def reduceToColIndex(df, col_name, head, mid, end):
    '''reduce soil type cols to single col with col index'''
    df_ = df.iloc[:, :head].join(df.iloc[:,head:end]                           .dot(range(1,mid)).to_frame(col_name))                           .join(df.iloc[:,end])
    return df_


# In[ ]:


#reducing Soil_Type cols to col index
X = reduceToColIndex(X, 'Soil_Type1', 14, 38, -1)
test = reduceToColIndex(test, 'Soil_Type1', 14, 38, -1)


# In[ ]:


from sklearn.model_selection import train_test_split
#splitting data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV


# In[ ]:


#preparing model
model = RandomForestClassifier()
params_rf = {'n_estimators': [10, 50, 75]}
rf_gs = GridSearchCV(model, params_rf, cv=5)
rf_gs.fit(X_train,y_train)


# In[ ]:


#get the error rate
val_predictions = rf_gs.predict(X_val)
val_mae = mean_absolute_error(y_val,val_predictions)
print('Fourth try mae with RFClassifier: ', val_mae)


# In[ ]:


test_preds = rf_gs.predict(test)
output = pd.DataFrame({'Id': testids, 'Cover_Type': test_preds.astype(int)})
output.to_csv('submission.csv', index=False)

