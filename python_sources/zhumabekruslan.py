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


train_ds = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
test_ds = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


# In[ ]:


train_ds.head()


# In[ ]:


train_ds.columns


# In[ ]:


train_ds.describe


# In[ ]:


train_ds.info()


# In[ ]:


train_ds.isna().sum()


# In[ ]:


train_ds.nunique()


# In[ ]:


X = train_ds.iloc[:, 2:].values
y = train_ds.iloc[:, 1].values


# In[ ]:


X[0]


# In[ ]:


y


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 80,stratify =y)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[ ]:


logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train, y_train)


# In[ ]:


y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[ ]:


test = test_ds.iloc[:, 1:].values
y_pred = logreg.predict(test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[ ]:


submission = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
submission['target'] = y_pred
submission.to_csv('logreg.csv', index=False)


# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


gnb = GaussianNB()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10, stratify = y)


# In[ ]:


gnb.fit(X_train, y_train)


# In[ ]:


y_pred = gnb.predict(X_test)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix

target_names = ['0', '1']

print(classification_report(y_test, y_pred, target_names=target_names))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 199998, random_state = 80,stratify =y)


# In[ ]:


y_pred = gnb.predict(test)


# In[ ]:


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('gaussian.csv', index=False)


# In[ ]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split


# In[ ]:


bnb = BernoulliNB()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10, stratify = y)


# In[ ]:


bnb.fit(X_train, y_train)
bnb.score(X_test, y_test)
y_pred = bnb.predict(X_test)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names))


# In[ ]:


y_pred = bnb.predict(test)


# In[ ]:


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('bernoulli.csv', index=False)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
from sklearn.tree import export_graphviz 


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10, stratify = y)


# In[ ]:


clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


y_pred = clf.predict(test)


# In[ ]:


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('decisionTree.csv', index=False)


# In[ ]:


import xgboost as xgb
import pandas as pd


# In[ ]:


xg_cl = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=1,
              learning_rate=0.1, max_delta_step=0, max_depth=8,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10, stratify = y)


# In[ ]:


xg_cl.fit(X_train, y_train)


# In[ ]:


y_pred = xg_cl.predict(X_test)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


print(classification_report(y_test, y_pred, target_names=target_names))


# In[ ]:


y_pred = xg_cl.predict(test)


# In[ ]:


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('xgboost.csv', index=False)


# In[ ]:


dataset_dmatrix = xgb.DMatrix(data = X,label = y)
dataset_dmatrix


# In[ ]:


params = {"objective":"reg:logistic", "max_depth":3}
params


# In[ ]:


cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "rmse", as_pandas = True, seed = 123)


# In[ ]:


print(cv_results)
print(1-cv_results["test-rmse-mean"].tail(1))


# In[ ]:


cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "auc", as_pandas = True, seed = 123)


# In[ ]:


print(cv_results["test-auc-mean"].tail(1))

