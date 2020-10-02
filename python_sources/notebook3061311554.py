#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


file_object = open('../input/gendermodel.py')
#myfirstforest = pd.read_csv('../input/myfirstforest.py')
try:
     gendermodel = file_object.readlines( )
finally:
     file_object.close( )
gendermodel


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
gendermodel = pd.read_csv('../input/gendermodel.csv')
genderclassmodel = pd.read_csv('../input/genderclassmodel.csv')


# In[ ]:


print(train.shape)
train_cols=train.columns
train_x_cols = train_cols.drop(['PassengerId','Name','Ticket'])
#print(set(train['Embarked']))
train_x=train[train_x_cols].fillna(99999).replace(['male','female','S','C','Q'],[1,0,1,2,3])
train_x['Cabin']=(pd.isnull(train['Cabin']))
#pd.merge(gendermodel,genderclassmodel,how='left',on='PassengerId')[a.Survived_x != a.Survived_y]


# In[ ]:


re
((train['Cabin'][train['Cabin'].notnull()])).replace(['C'],[1])


# In[ ]:


train = train_x
train_cols = train_x_cols.drop(['Survived'])


# In[ ]:


X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    train[train_cols], train['Survived'], test_size=0.4, random_state=0)
X_train.shape, y_train.shape,X_test.shape, y_test.shape


# In[ ]:


clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X_train.fillna(999), y_train)


# In[ ]:


clf.score(X_test.fillna(999), y_test)


# In[ ]:


clf.feature_importances_


# In[ ]:


train_cols


# In[ ]:




