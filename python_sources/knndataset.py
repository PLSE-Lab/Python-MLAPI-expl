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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


dataset_train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
dataset_test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


# In[ ]:


dataset_train.head()


# In[ ]:


dataset_test.head()


# In[ ]:


dataset_train.info()


# In[ ]:


X_train = dataset_train.iloc[:, dataset_train.columns != 'target'].values
y_train = dataset_train.iloc[:, 1].values
X_test = dataset_test.values


# In[ ]:


dataset_train.describe()


# In[ ]:



dataset_train.target.value_counts() 


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X_train[:,0] = le.fit_transform(X_train[:,0])
X_test[:,0] = le.fit_transform(X_test[:,0])
knn = KNeighborsClassifier(11)


# In[ ]:


knn.fit(X_train, y_train)


# In[ ]:


y_preds = knn.predict(X_test)


# In[ ]:


y_preds


# In[ ]:


pd.concat([dataset_test.ID_code, pd.Series(y_preds).rename('target')], axis = 1).to_csv('knn_submission.csv')


# In[ ]:


result_dataset = pd.concat([dataset_test.ID_code, pd.Series(y_preds).rename('target')], axis = 1)


# In[ ]:





# In[ ]:


result_dataset.target.value_counts()


# In[ ]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression


# In[ ]:


log_reg_cls = LogisticRegression()


# In[ ]:


log_reg_cls.fit(X_train, y_train)


# In[ ]:


y_preds_log_reg = log_reg_cls.predict(X_test)


# In[ ]:


y_preds_log_reg.shape


# In[ ]:


y_preds_log_reg


# In[ ]:


pd.concat([dataset_test.ID_code, pd.Series(y_preds_log_reg).rename('target')], axis = 1).to_csv('log_reg_submission.csv')


# In[ ]:


log_reg_dataset = pd.concat([dataset_test.ID_code, pd.Series(y_preds_log_reg).rename('target')], axis = 1)
log_reg_dataset.target.value_counts()


# In[ ]:





# In[ ]:


from sklearn import svm


# In[ ]:


clf = svm.SVC()
clf.fit(X_train, y_train)


# In[ ]:


clf_pred = clf.predict(X_test)


# In[ ]:


clf_pred


# In[ ]:


svm_dataset = pd.concat([dataset_test.ID_code, pd.Series(clf_pred).rename('target')], axis = 1)
svm_dataset.target.value_counts()


# In[ ]:


svm_dataset.to_csv('svm_submission.csv')


# In[ ]:




