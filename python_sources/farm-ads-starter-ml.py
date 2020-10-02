#!/usr/bin/env python
# coding: utf-8

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


farm_ads = pd.read_csv('/kaggle/input/farm-ads-binary-classification/farm-ads',sep = " ",error_bad_lines=False,header=None)
farm_vect = pd.read_csv('/kaggle/input/farm-ads-binary-classification/farm-ads-vect',sep = " ",error_bad_lines=False,header=None)


# In[ ]:


farm_ads


# In[ ]:


farm_vect


# In[ ]:


data = farm_ads.merge(farm_vect,right_index = True,left_index = True)


# In[ ]:


data


# In[ ]:


from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
import xgboost as xgb
from sklearn.ensemble import BaggingRegressor
import numpy as np 
import pandas as pd 
import random


# In[ ]:


le = preprocessing.LabelEncoder()
for name in data.columns:
    if data[name].dtypes == "O":
        print(name)
        data[name] = data[name].astype(str)
        le.fit(data[name])
        data[name] = le.transform(data[name])


# In[ ]:


data['target'] = data['0_x'] + data['0_y']


# In[ ]:


data = data.drop(['0_x','0_y'],axis = 1)


# In[ ]:


data


# In[ ]:


X = data.drop('target',axis = 1)
y = data['target']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


clf = RandomForestClassifier(n_estimators = 400,min_samples_split = 2,min_samples_leaf = 1,max_features= 'sqrt',max_depth =None,bootstrap= False)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import balanced_accuracy_score


# In[ ]:


predictions


# In[ ]:


balanced_accuracy_score(y_test, predictions)


# In[ ]:




