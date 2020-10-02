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


import numpy as np
import pandas as pd
df=pd.read_csv('../input/loan-predication/train_u6lujuX_CVtuZ9i (1).csv')


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dropna(how="any",inplace=True)


# In[ ]:


df=df.drop('Loan_ID',axis=1)


# In[ ]:


dependents=pd.get_dummies(df['Dependents'],drop_first=True)


# In[ ]:


dependents.head()


# In[ ]:


property_Area=pd.get_dummies(df['Property_Area'],drop_first=True)


# In[ ]:


property_Area.head()


# In[ ]:


gender=pd.get_dummies(df['Gender'],drop_first=True)


# In[ ]:


gender.head()


# In[ ]:


married=pd.get_dummies(df['Married'],drop_first=True)


# In[ ]:


married.head()


# In[ ]:


education=pd.get_dummies(df['Education'],drop_first=True)


# In[ ]:


education.head()


# In[ ]:


self_Employed=pd.get_dummies(df['Self_Employed'],drop_first=True)


# In[ ]:


self_Employed.head()


# In[ ]:


df=df.drop(['Property_Area','Gender','Married','Education','Self_Employed','Dependents'],axis=1)


# In[ ]:


df.head()


# In[ ]:


df=pd.concat([df,property_Area,gender,married,education,self_Employed,dependents],axis=1)


# In[ ]:



df.head()


# In[ ]:


df.info()


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[ ]:


X=df.drop(['Loan_Status'],axis=1)


# In[ ]:


Y=df['Loan_Status']


# In[ ]:



X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=np.random)


# In[ ]:


model=LogisticRegression()


# In[ ]:


model.fit(X_train,Y_train)


# In[ ]:


predictions=model.predict(X_test)


# In[ ]:



print(confusion_matrix(Y_test, predictions))
print(accuracy_score(Y_test, predictions))

