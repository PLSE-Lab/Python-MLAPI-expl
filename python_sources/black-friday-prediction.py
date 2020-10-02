#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


import numpy as np
import seaborn as sns


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/train.csv')


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


1


# In[ ]:


df.fillna(value=0,inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dtypes


# In[ ]:


df['Gender'].nunique()


# In[ ]:


Gender=pd.get_dummies(df['Gender'],drop_first=True)


# In[ ]:


df=pd.concat([df,Gender],axis=1)
df.drop('Gender',axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df['Stay_In_Current_City_Years'].unique()


# In[ ]:


def stay(x):
    if x=='4+':
        return 4
    else:
        return int(x)
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].apply(stay)


# In[ ]:


df['Age'].unique()


# In[ ]:


def age(x):
    if x=='0-17':
        return 0
    elif x=='18-25':
        return 1
    elif x=='26-35':
        return 2
    elif x=='36-45':
        return 3
    elif x=='46-50':
        return 4
    elif x=='51-55':
        return 5
    else:
        return 6
df['Age']=df['Age'].apply(age)


# In[ ]:


df.head()


# In[ ]:


df.drop(['User_ID','Product_ID'],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


def city(x):
    if x=='A':
        return 0
    elif x=='B':
        return 1
    else :
        return 2
df['City_Category']=df['City_Category'].apply(city)


# In[ ]:


df.head()


# In[ ]:


df.dtypes


# In[ ]:


def con(x):
    return int(x)
df['Product_Category_2']=df['Product_Category_2'].apply(con)
df['Product_Category_3']=df['Product_Category_3'].apply(con)


# In[ ]:


df.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x=df[['Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3','M']]


# In[ ]:


y=df['Purchase']


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)


# In[ ]:





# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm=LinearRegression()


# In[ ]:


lm.fit(x_train,y_train)


# In[ ]:


print(lm.intercept_)


# In[ ]:


lm.coef_


# In[ ]:


predictions=lm.predict(x_test)


# In[ ]:


from sklearn import metrics


# In[ ]:


metrics.mean_absolute_error(y_test,predictions)


# In[ ]:


metrics.mean_squared_error(y_test,predictions)


# In[ ]:


y_test


# In[ ]:


predictions

