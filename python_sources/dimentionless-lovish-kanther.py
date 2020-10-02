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


import numpy as np
import pandas as pd


# In[ ]:


train=pd.read_csv("/kaggle/input/credit-risk-modeling-case-study/CRM_TrainData.csv")
test=pd.read_csv("/kaggle/input/credit-risk-modeling-case-study/CRM_TestData.csv")


# In[ ]:


train.head()


# In[ ]:


train=train.drop(['Loan ID','Customer ID'],1)


# In[ ]:


train.isna().sum()


# In[ ]:


test.head()


# In[ ]:


test=test.drop(['Loan ID','Customer ID','Unnamed: 2'],1)


# In[ ]:


def rem(x):
    x=(str(x).replace("$",""))
    return float(x)

train['Monthly Debt']=train['Monthly Debt'].apply(rem)
test['Monthly Debt']=test['Monthly Debt'].apply(rem)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


from sklearn.preprocessing import LabelBinarizer, LabelEncoder
lb = LabelBinarizer()
train['Loan Status'] = lb.fit_transform(train['Loan Status'])


# In[ ]:


train['Credit Score']=train['Credit Score'].fillna((train['Credit Score'].mean()))
train['Annual Income']=train['Annual Income'].fillna((train['Annual Income'].mean()))
train['Months since last delinquent']=train['Months since last delinquent'].fillna((train['Months since last delinquent'].mean()))


# In[ ]:


train.isna().sum()


# In[ ]:


train.head()


# In[ ]:


train['Bankruptcies'].unique()


# In[ ]:


train['Tax Liens'].unique()


# In[ ]:


train['Bankruptcies']=train['Bankruptcies'].fillna(int((train['Bankruptcies'].mean())))
train['Tax Liens']=train['Tax Liens'].fillna(int(train['Tax Liens'].mean()))


# In[ ]:


train.isna().sum()


# In[ ]:


train['Years in current job'].unique()


# In[ ]:


import re
def getnum(a):
    a=(str(a).strip("years"))
    a=(str(a).replace("+",""))
    a=(str(a).strip("<"))
    return float(a)
    


# In[ ]:


train['Years in current job']=train['Years in current job'].apply(getnum)


# In[ ]:


train['Years in current job']=train['Years in current job'].fillna(int((train['Years in current job'].mean())))


# In[ ]:


train.rename(columns={"Maximum Open Credit":"Maximum_Open_Credit"},inplace=True)
test.rename(columns={"Maximum Open Credit":"Maximum_Open_Credit"},inplace=True)


# In[ ]:


q = pd.Series(train['Maximum_Open_Credit'])
train['Maximum_Open_Credit']=pd.to_numeric(q,errors="coerce")


# In[ ]:


a = pd.Series(test['Maximum_Open_Credit'])
test['Maximum_Open_Credit']=pd.to_numeric(a, errors="coerce")


# In[ ]:


train['Maximum_Open_Credit']=train['Maximum_Open_Credit'].fillna((train['Maximum_Open_Credit'].mean()))
test['Maximum_Open_Credit']=test['Maximum_Open_Credit'].fillna((test['Maximum_Open_Credit'].mean()))


# In[ ]:


train.dtypes


# In[ ]:


train.isna().sum()


# In[ ]:


test.isna().sum()


# In[ ]:


test['Credit Score']=test['Credit Score'].fillna((test['Credit Score'].mean()))
test['Annual Income']=test['Annual Income'].fillna((test['Annual Income'].mean()))
test['Months since last delinquent']=test['Months since last delinquent'].fillna((test['Months since last delinquent'].mean()))
test['Bankruptcies']=test['Bankruptcies'].fillna(int((test['Bankruptcies'].mean())))
test['Tax Liens']=test['Tax Liens'].fillna(int(test['Tax Liens'].mean()))
test['Years in current job']=test['Years in current job'].apply(getnum)
test['Years in current job']=test['Years in current job'].fillna(int((test['Years in current job'].mean())))


# In[ ]:


test.isna().sum()


# In[ ]:


train=pd.get_dummies(train,columns=['Home Ownership','Term','Purpose'])
test=pd.get_dummies(test,columns=['Home Ownership','Term','Purpose'])


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train_y=train['Loan Status']
train_x=train.drop('Loan Status',axis= 1)


# In[ ]:


RF = RandomForestClassifier()


# In[ ]:


RF.fit(train_x,train_y)


# In[ ]:


pred = RF.predict(test)


# In[ ]:


pred


# In[ ]:


test=pd.read_csv("/kaggle/input/credit-risk-modeling-case-study/CRM_TestData.csv")


# In[ ]:


y_predict=pd.DataFrame(test['Loan ID'])


# In[ ]:


y_predict['Loan Status']=pred


# In[ ]:


y_predict


# In[ ]:


y_predict.to_csv("Dimentionless_Lovish_Kanther.csv")


# In[ ]:




