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


#reading the dataset
r=pd.read_csv('/kaggle/input/Credit_default_dataset.csv')
r.head(10)


# In[ ]:


#ID is not needed so it will be dropped
r.drop(columns='ID',inplace=True)


# In[ ]:


#arranging the PAY columns in order by renaming them
r.rename(columns={'PAY_0':'PAY_1'},inplace=True)


# In[ ]:


r['EDUCATION'].value_counts()


# In[ ]:


#reducing the unrequired number of categories in education and marriage
r['EDUCATION']=r['EDUCATION'].map({0:4,1:1,2:2,3:3,4:4,5:4,6:4})
r['EDUCATION'].value_counts()


# In[ ]:


r['MARRIAGE'].value_counts()


# In[ ]:


r['MARRIAGE']=r['MARRIAGE'].map({0:3,1:1,2:2,3:3})
r['MARRIAGE'].value_counts()


# In[ ]:


#taking independent and dependent features
x=r.iloc[:,:-1]
y=r.iloc[:,-1]


# In[ ]:


#hyperparamter tuning for XGBOOST
params={
    'eta':[0.01,0.05,0.10,0.15,0.20,0.25],
    'max_depth':[3,4,5,6,7],
    'min_child_weight':[2,3,5,6,7],
    'colsample_bytree':[0.5,0.6,0,7,0.9],
    'gamma':[0.1,0.2,0.3,0.4]
}


# In[ ]:


#using randomized search cv for tuning
from sklearn.model_selection import RandomizedSearchCV as rs
import xgboost


# In[ ]:


#creating the classifier object
X=xgboost.XGBClassifier()


# In[ ]:


q=rs(X,param_distributions=params,n_iter=5,n_jobs=-1,scoring='accuracy',cv=5)
q.fit(x,y)


# In[ ]:


q.best_estimator_


# In[ ]:


z=xgboost.XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.9, eta=0.01, gamma=0.1,
              gpu_id=-1, importance_type='gain', interaction_constraints=None,
              learning_rate=0.00999999978, max_delta_step=0, max_depth=6,
              min_child_weight=3, missing=None, monotone_constraints=None,
              n_estimators=100, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)


# In[ ]:


from sklearn.model_selection import cross_val_score as cv
scores=cv(z,x,y,cv=10)


# In[ ]:


scores.mean()

