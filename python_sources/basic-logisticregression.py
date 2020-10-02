#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


y_train = train['target']
X_train = train.drop('target', axis=1)
X_train = X_train.drop('ID_code', axis=1)

X_test = test.drop('ID_code', axis = 1)


# In[ ]:


from sklearn import preprocessing
X_train = preprocessing.scale(X_train)


# In[ ]:


lr = LogisticRegression(class_weight=None, penalty='l2', C=0.001, solver='liblinear').fit(X_train, y_train)


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = lr.predict_proba(X_test)[:,1]
submission.to_csv('submission_LogisticRegression.csv', index=False)
submission.head(20)

