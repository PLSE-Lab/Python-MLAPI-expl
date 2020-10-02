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


# # First glimpse of the dataset

# In[ ]:


df = pd.read_csv('../input/heart-disease-prediction-using-logistic-regression/framingham.csv')
df.head()


# In[ ]:


print("The dataset is consisted of {} entries and {} features".format(df.shape[0], df.shape[1]))


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# # Filling missing data

# I decided to drop education because it seems not related to the target we want to predict

# In[ ]:


df = df.drop(['education'],axis=1)


# In[ ]:


df.cigsPerDay.describe()


# In[ ]:


df['cigsPerDay'].fillna(df['cigsPerDay'].median(), inplace=True)


# In[ ]:


df.BPMeds.value_counts()


# In[ ]:


df['BPMeds'].fillna(df['BPMeds'].value_counts().index[0], inplace=True)


# In[ ]:


df.totChol.describe()


# In[ ]:


df['totChol'].fillna(df['totChol'].mean(), inplace=True)


# In[ ]:


df.BMI.describe()


# In[ ]:


df['BMI'].fillna(df['BMI'].mean(), inplace=True)


# In[ ]:


df.heartRate.describe()


# In[ ]:


df['heartRate'].fillna(df['heartRate'].value_counts().index[0], inplace=True)


# In[ ]:


df.glucose.describe()


# In[ ]:


df['glucose'].fillna(df['glucose'].mean(), inplace=True)


# In[ ]:


df.isnull().sum()


# # Fitting Default Model

# I first fit the data into logistic regression model without modifying any of its features

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from statsmodels.tools import add_constant
import warnings
warnings.filterwarnings('ignore')

X = df.drop(['TenYearCHD'], axis=1)
X = add_constant(X)
y = df['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

logReg = LogisticRegression().fit(X_train, y_train)

train_pred = logReg.predict(X_train)
test_pred = logReg.predict(X_test)

print('Train set accuracy score:', accuracy_score(y_train, train_pred))
print('Test set accuracy score:', accuracy_score(y_test, test_pred))


# # Backward Elimination

# I'm using backward elimination to select a certain amount of features and drop the rest of them to see if the model can perform better with new dataset

# In[ ]:


import statsmodels.api as sm

result = sm.Logit(y, X).fit()
result.summary()


# We need to drop one feature with the highest p-value among those above 0.05 (5%), and continue this process until there is no feature which its p-value is higher than 0.05.

# In[ ]:


X.drop(['currentSmoker'], axis=1, inplace=True)

result = sm.Logit(y, X).fit()
result.summary()


# In[ ]:


X.drop(['BMI'], axis=1, inplace=True)

result = sm.Logit(y, X).fit()
result.summary()


# In[ ]:


X.drop(['heartRate'], axis=1, inplace=True)

result = sm.Logit(y, X).fit()
result.summary()


# In[ ]:


X.drop(['diaBP'], axis=1, inplace=True)

result = sm.Logit(y, X).fit()
result.summary()


# In[ ]:


X.drop(['diabetes'], axis=1, inplace=True)

result = sm.Logit(y, X).fit()
result.summary()


# In[ ]:


X.drop(['BPMeds'], axis=1, inplace=True)

result = sm.Logit(y, X).fit()
result.summary()


# In[ ]:


X.drop(['prevalentHyp'], axis=1, inplace=True)

result = sm.Logit(y, X).fit()
result.summary()


# In[ ]:


X.drop(['totChol'], axis=1, inplace=True)

result = sm.Logit(y, X).fit()
result.summary()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

logReg = LogisticRegression().fit(X_train, y_train)
train_pred = logReg.predict(X_train)
test_pred = logReg.predict(X_test)

print('New train set accuracy:', accuracy_score(y_train, train_pred))
print('New test set accuracy:', accuracy_score(y_test, test_pred))


# In[ ]:


confusion_matrix(y_test, test_pred)


# After performing backward elimination, there is no significant improvement in prediction accuracy, either training and test set. The problem could be the way I performed backward elimination with the feature chosen. 
