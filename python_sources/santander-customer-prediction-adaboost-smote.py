#!/usr/bin/env python
# coding: utf-8

# ![](http://investtribune.com/wp-content/uploads/logos/Logos/BSMX.png)

# ## Santander Customer Transaction Prediction ##

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE


# In[ ]:


df_train = pd.read_csv("../input/train.csv")


# In[ ]:


df_train.head()


# In[ ]:


print("Number of samples in dataset -", len(df_train))


# In[ ]:


# Checking for imbalance in dataset
print("Size of target 0 -" ,len(df_train[df_train['target'] == 0]))
print("Size of target 1 -" ,len(df_train[df_train['target'] == 1]))


# In[ ]:


y_train = df_train['target']


# In[ ]:


df_train = df_train.drop(['ID_code','target'], axis=1)


# In[ ]:


df_train.head()


# In[ ]:


sm = SMOTE(ratio='minority',random_state=9, n_jobs=10)


# In[ ]:


df_train_sm, y_train_sm = sm.fit_sample(df_train, y_train)


# In[ ]:


len(df_train_sm)


# In[ ]:


df_train = pd.DataFrame(df_train_sm)


# In[ ]:


y_train = pd.DataFrame(y_train_sm)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_train, y_train, test_size=0.2, random_state=42)


# In[ ]:


model = AdaBoostClassifier(n_estimators=200,learning_rate=0.5)


# In[ ]:


model.fit(x_train, y_train)


# In[ ]:


y_pred = model.predict(x_test)


# In[ ]:


y_pred


# In[ ]:


ada = model.feature_importances_


# In[ ]:


from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)


# In[ ]:





# In[ ]:


"""
from scipy.stats import pearsonr
df_train['var_0'].corr(df_train['var_6'])
model = AdaBoostClassifier(n_estimators=200,learning_rate=0.5)
svc=SVC(probability=True, kernel='linear')
"""


# In[ ]:


"""
df_test = pd.read_csv("../input/test.csv")
df_test.head()
test_id = df_test['ID_code']
df_test = df_test.drop(['ID_code'], axis=1)
predictions = model.predict(df_test)
test = test_id.to_frame(name='ID_code')
test['target'] = pd.Series(predictions)
test.to_csv("submission.csv", columns = test.columns, index=False)
"""

