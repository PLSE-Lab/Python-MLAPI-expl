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


sample_submission = pd.read_csv('/kaggle/input/kave-hackathonv2/sample_submission.csv')
test = pd.read_csv('/kaggle/input/kave-hackathonv2/test.csv')
train = pd.read_csv('/kaggle/input/kave-hackathonv2/train.csv')


# In[ ]:


from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


# In[ ]:


train


# In[ ]:


train.info()


# In[ ]:


label_encoder = preprocessing.LabelEncoder() 

train['Class'] = label_encoder.fit_transform(train['Class']) 
train['Class'].unique()


# In[ ]:


columns = train.columns[:-1]
for col in columns:
    train[col] = pd.to_numeric(train[col], errors='coerce')
    
columns = test.columns
for col in columns:
    test[col] = pd.to_numeric(test[col], errors='coerce')


# In[ ]:


columns = train.columns[:-1]
for col in columns:
    train[col] = train[col].fillna(train[col].mean())
    
columns = test.columns
for col in columns:
    test[col] = test[col].fillna(test[col].mean())


# In[ ]:


train = train.drop(['ID'], axis = 1)
test = test.drop(['ID'], axis = 1)


# In[ ]:


x = train.iloc[:,0:31]
y = train.iloc[:,-1]


# In[ ]:


# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# In[ ]:


# model = lgb.LGBMClassifier()
# model.fit(X_train, y_train)
# print(model)


# In[ ]:


# expected_y  = y_test
# predicted_y = model.predict(X_test)


# In[ ]:


# from sklearn import metrics

# print(); print(metrics.classification_report(expected_y, predicted_y))
# print(); print(metrics.confusion_matrix(expected_y, predicted_y))


# In[ ]:


data = train

X = data.iloc[:,0:31]
y = data.iloc[:,-1]

model = ExtraTreesClassifier()
model.fit(X,y)

feature_importance_normalized = np.std([tree.feature_importances_ for tree in 
                                        model], 
                                        axis = 0)

accuracy_score(y, model.predict(X))


# In[ ]:


y_pred = model.predict(test)
submission = test.copy()

submission['ID'] = submission.index
submission['Class'] = y_pred
submission = submission[['ID','Class']]
submission.Class.value_counts()
submission['Class'] = submission.Class.astype(int)
submission['Class'] = submission['Class'].map({0:"Bitkisel", 1:"Hayvansal", 2:"Mix"})
submission


# In[ ]:


submission.to_csv('isolation_final.csv',index=False)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


from sklearn.metrics import classification_report
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

