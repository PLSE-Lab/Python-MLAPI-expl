#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cross_validation import KFold, train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
import xgboost as xgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


read_file=pd.read_csv('../input/train.csv',  sep=",")
read_file.head(5)


# In[ ]:


for i in read_file.columns:
    if(not i.find('cat')):
        categories={}
        for p,j in enumerate(read_file[i].unique()):
            categories[j]=p
        read_file[i]=read_file[i].map(categories)
read_file=read_file.drop(['id'], axis=1)


# In[ ]:


data=read_file[:500]
clf=RandomForestRegressor( n_estimators=22)
clf.fit(data.ix[:, data.columns != 'loss'], data['loss'])


# In[ ]:


data=read_file[0:100]
clf.predict(data.ix[:, data.columns != 'loss'])


# In[ ]:


training_data, validation_data=train_test_split(read_file, test_size=0.30, random_state=42)


# In[ ]:


class Sklearn_class(object):
    def __init__(self,clf, seed, params=0):
        params['random_state']=seed
        self.clf=clf(**params)
    def train(self, train_x, train_y):
        cv=KFold(len(train_x), n_folds=10)
        for train, validation in cv:
            training=self.clf.fit(train_x.iloc[train], train_y.iloc[train])
            validation=training.predict(train_x.iloc[validation])
            print(mean_absolute_error(validation, train_y.iloc[validation]))
    def test(self, y_test):
        return self.clf.predict(y_test)
        


# In[ ]:


et_params = {
    'n_jobs': 16,
    'n_estimators': 200,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 16,
    'n_estimators': 200,
    'max_features': 0.2,
    'max_depth': 8,
    'min_samples_leaf': 2,
}

et = Sklearn_class(clf=ExtraTreesRegressor, seed=0, params=et_params)
train=et.train(training_data.ix[:, training_data.columns != 'loss'], training_data['loss'])
results=et.test(validation_data.ix[:, validation_data.columns != 'loss'])


# In[ ]:


rfr = Sklearn_class(clf=RandomForestRegressor, seed=0, params=et_params)
train=rfr.train(training_data.ix[:, training_data.columns != 'loss'], training_data['loss'])
results_rf=rfr.test(validation_data.ix[:, validation_data.columns != 'loss'])


# In[ ]:


meanvalidation_data.ix[:, validation_data.columns != 'loss']_absolute_error(results, validation_data['loss']), mean_absolute_error(results_rf, validation_data['loss'])


# In[ ]:


dtrain = xgb.DMatrix(training_data.ix[:, training_data.columns != 'loss'], training_data['loss'])
xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.075,
    'objective': 'reg:linear',
    'max_depth': 7,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'mae',
    'nrounds': 250
}
num_round = 250
bst = xgb.train(xgb_params, dtrain, num_round)


# In[ ]:


dtest=xgb.DMatrix(validation_data.ix[:, validation_data.columns != 'loss'])
results=bst.predict(dtest)


# In[ ]:


mean_absolute_error(results, validation_data['loss'])

