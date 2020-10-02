#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble


# In[ ]:


train = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_treino.csv')
test = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_teste.csv')


# In[ ]:


target = train['target'].values
train = train.drop(['ID','target'],axis=1)


# In[ ]:


id_test = test['ID'].values
test = test.drop(['ID'],axis=1)


# In[ ]:


for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_series.dtype == 'O':
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
    else:
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            train.loc[train_series.isnull(), train_name] = -9999
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -9999


# In[ ]:


X_train = train
X_test = test


# In[ ]:


extc = ExtraTreesClassifier(n_estimators =700,
                            max_features = 50,
                            criterion = 'entropy',
                            min_samples_split = 5,
                            max_depth = 50,
                            min_samples_leaf = 5,
                            random_state = 420) 


# In[ ]:


extc.fit(X_train,target)


# In[ ]:


y_pred = extc.predict_proba(X_test)


# In[ ]:


pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:,1]}).to_csv('submission_extc_clean.csv',index=False)

