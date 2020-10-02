#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv('/kaggle/input/loan-predication/train_u6lujuX_CVtuZ9i (1).csv')


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.dtypes


# In[ ]:


df.boxplot(column='ApplicantIncome', by='Education')


# In[ ]:


df['ApplicantIncome'].hist (bins=50)


# In[ ]:


df['LoanAmount'].hist (bins=50)


# In[ ]:


df.apply(lambda X: sum(X.isnull()),axis=0)


# In[ ]:


df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)


# In[ ]:


df.apply(lambda X: sum(X.isnull()),axis=0)


# In[ ]:


df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(),inplace=True)


# In[ ]:


df.apply(lambda X: sum(X.isnull()),axis=0)


# In[ ]:


df['Credit_History'].fillna(df['Credit_History'].mean(),inplace=True)


# In[ ]:


df.apply(lambda X: sum(X.isnull()),axis=0)


# In[ ]:


df.mean()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics


# In[ ]:


X= df.iloc[:,[8,10]].values
y=df.iloc[:, 12].values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25, random_state=0)


# In[ ]:


from sklearn import preprocessing 
from sklearn.StandardScaler(
SC_X= StandardScalar()


# In[ ]:


X_train = SC_X.fit_transform(X_train)
X_test = SC_X.transform(X_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


# In[ ]:


y_pred=classifier.predict(X_test)
y_pred


# In[ ]:


from sklearn.metrics import confusion_matrix
cm= confusion_matrix (y_test,y_pred)
cm


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score


# In[ ]:




