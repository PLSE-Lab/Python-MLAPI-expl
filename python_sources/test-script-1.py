#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import VotingClassifier
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/train.csv')
y = data['TARGET'].values
X = data.drop(['ID','TARGET'], axis=1).values
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.9)

# Any results you write to the current directory are saved as output.


# In[ ]:


#X_train = X_train[:size]
#y_train = y_train[:size]
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression()
clf2 = svm.SVC(probability=True)

vc = VotingClassifier(estimators=[('RandomForest', clf1), ('SVC', clf2)])
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)
vc.fit(X_train, y_train)


# In[ ]:


vc.


# In[ ]:


estimators = [('reduce_dim', SelectKBest(f_classif, k=32)), ('svm', svm.SVC(probability=True))]
#pipeline = Pipeline(estimators)
#size = 1000
#pipeline.fit(X_train, y_train)
print('AUC:', roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1]))

