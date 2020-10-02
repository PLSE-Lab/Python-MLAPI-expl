#!/usr/bin/env python
# coding: utf-8

# # In this notebook I'm going to predict employee retention using logistic regression

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv('../input/HR_comma_sep.csv')


# #  Data exploration and visualization

# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.groupby('left').count()['satisfaction_level']


# In[ ]:


df.groupby('left').mean()


# In[ ]:


fig,axis=plt.subplots(nrows=2,ncols=1,figsize=(12,10))
sns.countplot(x='salary',hue='left',data=df,ax=axis[0])
sns.countplot(x='Department',hue='left',data=df,ax=axis[1])


# <h3 style="color:blue">From the data analysis so far we can conclude that we will use following variables as dependant variables in our model</h3>
# <ol>
#     <li>**Satisfaction Level**</li>
#     <li>**Average Monthly Hours**</li>
#     <li>**Promotion Last 5 Years**</li>
#     <li>**Salary**</li>
# </ol>

# In[ ]:


df.columns


# # Model selection

# In[ ]:


subdf=df[['satisfaction_level', 'average_montly_hours',  'Work_accident',
          'promotion_last_5years', 'salary']]


# In[ ]:


subdf.head()


# In[ ]:


dummies=pd.get_dummies(subdf['salary']) 


# In[ ]:


dummies.head()


# In[ ]:


dffinal=pd.concat([subdf,dummies],axis='columns')
dffinal.head(3)


# In[ ]:


X=dffinal.drop('salary',axis='columns')
y=df[['left']]
X.head(3)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# # Training model

# In[ ]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


model.coef_


# # Prediction

# In[ ]:


model.predict(x_test)


# In[ ]:


model.predict_proba(x_test)


# In[ ]:


model.score(x_test,y_test)

