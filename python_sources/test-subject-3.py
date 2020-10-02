#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


df=pd.read_csv('../input/loan-predication/train_u6lujuX_CVtuZ9i (1).csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


df=df.drop('Loan_ID',axis=1)


# In[ ]:


df.dropna(how='any',inplace=True)


# In[ ]:


sns.heatmap(df.corr())


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer


# In[ ]:


df1=pd.get_dummies(df['Gender'],drop_first=True)
df=pd.concat([df1,df],axis=1)


# In[ ]:


df2=pd.get_dummies(df['Married'],drop_first=True)
df=pd.concat([df2,df],axis=1)


# In[ ]:


df3=pd.get_dummies(df['Dependents'],drop_first=True)
df=pd.concat([df3,df],axis=1)


# In[ ]:


df4=pd.get_dummies(df['Self_Employed'],drop_first=True)
df=pd.concat([df4,df],axis=1)


# In[ ]:


df5=pd.get_dummies(df['Education'],drop_first=True)
df=pd.concat([df5,df],axis=1)


# In[ ]:


df6=pd.get_dummies(df['Property_Area'],drop_first=True)
df=pd.concat([df6,df],axis=1)


# In[ ]:


df.head()


# In[ ]:


df=df.drop(['Gender','Dependents','Education','Self_Employed','Property_Area','Married'],axis=1)


# In[ ]:


df.head()


# In[ ]:


x=df.drop(['Loan_Status'], axis=1)


# In[ ]:


y=df['Loan_Status']


# In[ ]:


y.head()


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=1)


# In[ ]:


model=LogisticRegression()


# In[ ]:


model.fit(x_train,y_train)


# In[ ]:


predictions=model.predict(x_test)


# In[ ]:


print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))


# In[ ]:




