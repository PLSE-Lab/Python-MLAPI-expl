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


# In[ ]:


train = pd.read_csv('/kaggle/input/credit-risk-modeling-case-study/CRM_TrainData.csv')
train.head()


# In[ ]:


test = pd.read_csv('/kaggle/input/credit-risk-modeling-case-study/CRM_TestData.csv')
test.head()


# In[ ]:


train['Loan Status'].unique()


# In[ ]:


train = train.drop(['Loan ID','Customer ID'],1)
train.head()


# In[ ]:


train.shape


# In[ ]:


train.isna().sum()


# In[ ]:


train['Credit Score'].fillna((train['Credit Score'].mean()),inplace=True)
train['Annual Income'].fillna((train['Annual Income'].mean()),inplace=True)
train['Months since last delinquent'].fillna((train['Months since last delinquent'].mean()),inplace=True)
train['Bankruptcies'].fillna((train['Bankruptcies'].mean()),inplace=True)
train['Tax Liens'].fillna((train['Tax Liens'].mean()),inplace=True)


# In[ ]:


train.isna().sum()


# In[ ]:


def strip(a):
    a=(str(a).strip("years"))
    a=(str(a).replace("+",""))
    a=(str(a).strip("<"))
    return float(a)


# In[ ]:


train['Years in current job']=train['Years in current job'].apply(strip)


# In[ ]:


train['Years in current job'].fillna((train['Years in current job'].mean()),inplace=True)


# In[ ]:


train.isna().sum()


# In[ ]:


test = test.drop(['Loan ID','Customer ID','Unnamed: 2'],1)
test.head()


# In[ ]:


test.isna().sum()


# In[ ]:


test['Credit Score'].fillna((test['Credit Score'].mean()),inplace=True)
test['Annual Income'].fillna((test['Annual Income'].mean()),inplace=True)
test['Months since last delinquent'].fillna((test['Months since last delinquent'].mean()),inplace=True)
test['Bankruptcies'].fillna((test['Bankruptcies'].mean()),inplace=True)
test['Tax Liens'].fillna((test['Tax Liens'].mean()),inplace=True)


# In[ ]:


test['Years in current job']=test['Years in current job'].apply(strip)


# In[ ]:


test['Years in current job'].fillna((test['Years in current job'].mean()),inplace=True)


# In[ ]:


test.isna().sum()


# In[ ]:


train.dtypes


# In[ ]:


train = pd.get_dummies(train, columns=['Term','Home Ownership','Purpose'],drop_first=True)
train.head()


# In[ ]:


train.shape


# In[ ]:


test = pd.get_dummies(test, columns=['Term','Home Ownership','Purpose'],drop_first=True)
test.head()


# In[ ]:


test.shape


# In[ ]:


train= pd.get_dummies(train, columns=['Loan Status'],drop_first=True)
train.head()


# In[ ]:


train.shape


# In[ ]:


def dollar(a):
    a=(str(a).replace("$"," "))
    return a


# In[ ]:


train['Monthly Debt'] = train['Monthly Debt'].apply(dollar)


# In[ ]:


test['Monthly Debt'] = test['Monthly Debt'].apply(dollar)


# In[ ]:


x = pd.Series(train['Maximum Open Credit'])
train['Maximum Open Credit']=pd.to_numeric(x, errors='coerce')


# In[ ]:


y = pd.Series(test['Maximum Open Credit'])
test['Maximum Open Credit']=pd.to_numeric(y, errors='coerce')


# In[ ]:


train['Maximum Open Credit'].fillna((train['Maximum Open Credit'].mean()),inplace=True)
test['Maximum Open Credit'].fillna((test['Maximum Open Credit'].mean()),inplace=True)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


lr = LogisticRegression()
gnb = GaussianNB()
rfr = RandomForestClassifier()


# In[ ]:


x_train=train.drop('Loan Status_Fully Paid',axis= 1)
y_train=train['Loan Status_Fully Paid']


# In[ ]:


rfr.fit(x_train,y_train)


# In[ ]:


y_pred = rfr.predict(test)


# In[ ]:


y_pred


# In[ ]:


test = pd.read_csv("/kaggle/input/credit-risk-modeling-case-study/CRM_TestData.csv")


# In[ ]:


y_pred_rf=pd.DataFrame(test['Loan ID'])


# In[ ]:


y_pred_rf


# In[ ]:


y_pred_rf['Loan Status']=y_pred


# In[ ]:


y_pred_rf


# In[ ]:


y_pred_rf.to_csv("Devesh-Patodia-Dimensionless-Technologies-1.csv")


# In[ ]:




