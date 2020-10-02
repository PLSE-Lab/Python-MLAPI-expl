#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')


# In[ ]:


df.head()


# In[ ]:


df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
# Maybe we can drop CreditScore


# In[ ]:


# EDA
df.describe()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)


# In[ ]:


df.head()


# In[ ]:


# Let's select all our categorical columns
x = OneHotEncoder(sparse=False, drop='first')

encoded = x.fit_transform(df[['Geography', 'Gender']])


# In[ ]:


encoded = pd.DataFrame(encoded, columns=x.get_feature_names())


# In[ ]:


len(encoded), len(df)


# In[ ]:


# pd.concat([df, encoded], axis=1, join=[df.index])

# pd.merge(df,encoded, on=encoded.index)

df = df.reset_index()
encoded = encoded.reset_index()


# In[ ]:


pd.concat([df, encoded], axis=1)


# In[ ]:


df = pd.get_dummies(df,columns=['Geography', 'Gender'], drop_first=True)


# In[ ]:


df.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


df.drop('index', axis=1, inplace=True)


# In[ ]:


X = df.loc[:, df.columns != 'Exited']
y = df.Exited


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


s = StandardScaler()
X_train_s = s.fit_transform(X_train)
X_test_s  = s.transform(X_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


m = LogisticRegression()


# In[ ]:


m.fit(X_train_s, y_train)
y_pred = m.predict(X_test_s)


# In[ ]:


m.score(X_test, y_test) # Ok this seems to be a low number


# In[ ]:


ver = pd.DataFrame([y_pred, y_test]).transpose()#.columns=['ypred', 'y_test']


# In[ ]:


ver.columns=['ypred', 'y_test']


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_test, y_pred)


# In[ ]:


print('Accuracy Score: ', ((1543+79) * 100 / 2000), '%') # Before I remembered there's an actual module to do this for me


# In[ ]:


# Let's see if we're even allowed to use accuracy. best to use when there's a more even distribution
df.Exited.value_counts()


# In[ ]:


len(df)


# In[ ]:


df.dtypes


# In[ ]:


import lightgbm as lgb


# In[ ]:


clf = lgb.LGBMClassifier()


# In[ ]:


clf.fit(X_train_s, y_train)
y_pred_lgb = clf.predict(X_test_s)


# In[ ]:


confusion_matrix(y_test, y_pred_lgb)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test, y_pred_lgb) # Best scenario so far


# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


xg_m = XGBClassifier()


# In[ ]:


xg_m.fit(X_train_s, y_train)
y_pred_xg = xg_m.predict(X_test_s)


# In[ ]:


accuracy_score(y_test, y_pred_xg)

