#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as n
import matplotlib.pyplot as ma
import seaborn as se


# In[ ]:


df=pd.read_csv('heart.csv')
df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.groupby('target').mean()


# In[ ]:


df=df.drop(['thal','fbs','trestbps'],1)
df.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


n.random.seed(21)
x_train,x_test,y_train,y_test=train_test_split(df[['age','sex','cp','chol','restecg','thalach','exang','oldpeak','slope','ca']],df.target,train_size=0.77,random_state=123)


# In[ ]:


x_train.head()


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


mo=LogisticRegression()
mo.fit(x_train,y_train)


# In[ ]:


mo.score(x_test,y_test)


# In[ ]:


mo.predict_proba(x_test)


# In[ ]:




