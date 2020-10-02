#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


data = pd.read_csv('/kaggle/input/credit-risk-modeling-case-study/CRM_TrainData.csv')


# In[ ]:


data.head()


# In[ ]:


df = data.drop(['Loan ID','Customer ID','Months since last delinquent'], axis = 1)


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.dtypes


# In[ ]:


(df['Number of Credit Problems'].value_counts())/df.shape[0]


# In[ ]:


df['Monthly Debt'] = pd.to_numeric(df['Monthly Debt'].astype(str).str.replace('$',''), errors='coerce').fillna(0).astype(float)


# In[ ]:


df['Maximum Open Credit'] = df['Maximum Open Credit'].astype('float64')


# In[ ]:


df.dtypes


# In[ ]:


X = df.drop('Loan Status', axis = 1)
Y = df['Loan Status']


# In[ ]:


X = pd.get_dummies(X,drop_first=True)


# In[ ]:


X.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
cols = list(X)
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=cols)


# In[ ]:


X.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


rfc = RandomForestClassifier()
params = {'n_estimators':[50,100,150,200],
          'criterion' : ['gini', 'entropy'],
          'max_depth' : [5,10,15,20],
          'max_features': ['log2', 'sqrt']
         }
cv_rfc = GridSearchCV(estimator=rfc, param_grid = params, cv = 5, n_jobs= -1)
cv_rfc.fit(X,Y)


# In[ ]:


cv_rfc.best_score_


# In[ ]:


cv_rfc.best_params_


# In[ ]:


test = pd.read_csv(r'C:\Users\harsh\Documents\MPSTME\Internship\Dimensionless\CRM_TestData.csv')


# In[ ]:


test.shape


# In[ ]:


test_X = test.drop(['Customer ID','Months since last delinquent'], axis = 1)


# In[ ]:


test_X.isna().sum()


# In[ ]:


test_X['Years in current job'].mode()


# In[ ]:


test_X['Credit Score'].fillna((test_X['Credit Score'].mean()), inplace=True)
test_X['Annual Income'].fillna((test_X['Annual Income'].mean()), inplace=True)
test_X['Bankruptcies'].fillna((test_X['Bankruptcies'].mean()), inplace=True)
test_X['Tax Liens'].fillna((test_X['Tax Liens'].mean()), inplace=True)
test_X['Years in current job'].fillna('10+ years', inplace=True)


# In[ ]:


test_X.to_csv('Submission2.csv')


# In[ ]:


test_X.dtypes


# In[ ]:


test_X.shape


# In[ ]:


test_X['Monthly Debt'] = pd.to_numeric(test_X['Monthly Debt'].astype(str).str.replace('$',''), errors='coerce').fillna(0).astype(float)


# In[ ]:


test_X['Maximum Open Credit'] = pd.to_numeric(test_X['Maximum Open Credit'].astype(str).str.replace('#VALUE!',''), errors='coerce').fillna(0).astype(float)


# In[ ]:


test_X.dtypes


# In[ ]:


test_X1 = pd.get_dummies(test_X.iloc[:,1:], drop_first=True)


# In[ ]:


cols = list(test_X1)
test_X1 = scaler.transform(test_X1)
test_X1 = pd.DataFrame(test_X1, columns=cols)


# In[ ]:


test_X1['Loan Status'] = cv_rfc.predict(test_X1)


# In[ ]:


test_X1['Loan Status'].value_counts()


# In[ ]:


test_X.head()


# In[ ]:


test_X1.to_csv('Submission2.csv')

