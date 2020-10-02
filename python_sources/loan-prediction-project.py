#!/usr/bin/env python
# coding: utf-8

# # Importing packages

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# **Reading train data**

# In[ ]:


df=pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# # To check null values

# In[ ]:


df.isnull().sum()


# In[ ]:


df.fillna(df.mean(),inplace=True)


# In[ ]:


df.isnull().sum()


# **dropping rows with null object**

# In[ ]:


df.dropna(how='any',inplace=True)


# In[ ]:


df.info()


# In[ ]:


df.head()


# **Dropping ID**

# In[ ]:


df=df.drop(['Loan_ID'],axis=1)


# # *creating dummy variables for categorial features*

# In[ ]:


a=df.select_dtypes('object').columns[:-1]


# leaving the last column cause as it is bcoz its our dependent variable we cannot encode it

# In[ ]:


a


# **these will be encoded**

# In[ ]:


df1=pd.DataFrame()


# In[ ]:


for i in a:
    df2=pd.get_dummies(df[i],drop_first=True)
    df1=pd.concat([df2,df1],axis=1)
    df=df.drop(i,axis=1)


# In[ ]:


df=pd.concat([df1,df],axis=1)


# In[ ]:


df.head()


# In[ ]:


df.info()


# # *CREATING TRAIN TEST SPLIT*

# In[ ]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[ ]:


x.head()


# In[ ]:


y.head()


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[ ]:


model=LogisticRegression()


# In[ ]:


model.fit(x_train,y_train)


# In[ ]:


y_pred=model.predict(x_test)


# In[ ]:


print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

