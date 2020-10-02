#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams["figure.figsize"] =(20,10)


# In[ ]:


get_ipython().system('ls')


# In[ ]:


df = pd.read_csv('../input/housing-in-london/housing_in_london_monthly_variables.csv')


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.shape


# In[ ]:


df.isna().sum()


# In[ ]:


#droping the no_of_crimes column as it has no meaningful data
df1 = df.drop(['no_of_crimes'],axis='columns')


# In[ ]:


df1.head()


# In[ ]:


#checking if we have NA values
df1.isna().sum()


# In[ ]:


df1.shape


# In[ ]:


#droping the 94 columns of houses sold na values
df2 = df1.dropna()


# In[ ]:


df2.isna().sum()


# In[ ]:


df2.shape


# In[ ]:


df2.head()


# In[ ]:


import datetime


# In[ ]:


#checking the type of date, it is an object i.e string which needs to be converted to datetime
df2['date'].dtype


# In[ ]:


#checking the type of all the columns
df2.dtypes


# In[ ]:


#Copying the dataframe into a new df
df3 = df2.copy()


# In[ ]:


df3.shape


# In[ ]:


df2.shape


# In[ ]:


#Converting the string ino the datetime
df3['datetime'] = pd.to_datetime(df3['date']) 


# In[ ]:


df3.head()


# In[ ]:


df3.dtypes


# In[ ]:


#dropping the date with string values.
df3.drop(['date'],axis='columns')


# In[ ]:


#let us check what are the unique values for area
df3['area'].unique()


# In[ ]:


#let us check what are the unique values for area
df3['houses_sold'].unique()


# In[ ]:


matplotlib.rcParams["figure.figsize"] =(30,20)
matplotlib.pyplot.bar(df3['area'],df3['average_price'])


# In[ ]:


def which_city(str):
    if 'london' in str:
        return 'London'
    else:
        return 'Rest of England'

which_city('east london')


# In[ ]:


#feature engineering for a new feature called city
df3['city'] =  df3['area'].apply(lambda x: which_city(x))


# In[ ]:


df3.head()


# In[ ]:


matplotlib.pyplot.bar(df3['city'],df3['average_price'])


# In[ ]:


#new feature called year

df3['year'] =  df3['datetime'].apply(lambda x: x.year)


# In[ ]:


df3.head()


# In[ ]:


df3.drop(['date'],axis='columns')


# In[ ]:


df4 = df3.copy()
df5 = df4.groupby('year').sum()


# In[ ]:


df5.head()


# In[ ]:



df5['houses_sold'].plot(kind='bar')


# In[ ]:


df5['average_price'].plot(secondary_y=True)


# In[ ]:


df4.head()


# In[ ]:


df4.shape


# In[ ]:


df6 = df4.copy()


# In[ ]:


df6['revenue'] = df6['average_price']*df6['houses_sold']


# In[ ]:


df6.head()


# In[ ]:


rev = df6.groupby('year').sum()


# In[ ]:


rev.head()


# In[ ]:


rev['revenue'].plot(kind="bar")


# In[ ]:


import sklearn


# In[ ]:


df4.head()


# In[ ]:


df4.info()
#to run a ML we need to conver objects into non-object notations


# In[ ]:


df4['code'].unique()


# In[ ]:


df4['code'] = df.code.str.replace('E','').astype(float)


# In[ ]:


df4['code'].unique()


# In[ ]:


df5 = df4.drop(['date','borough_flag'],axis='columns')


# In[ ]:


df5['area'].unique()


# In[ ]:


area_dummies = pd.get_dummies(df5['area'])


# In[ ]:


df7 = pd.concat([df5,area_dummies],axis='columns')


# In[ ]:


df7.describe()


# In[ ]:


df8 = df7.drop(['area','city','year','datetime'],axis='columns')


# In[ ]:


#a town or district which is an administrative unit.
X=df8.drop(['average_price'],axis='columns')
y=df8[['average_price']]


# In[ ]:


X.dtypes


# In[ ]:


y.dtypes


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=1,)


# In[ ]:


from sklearn.linear_model  import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)


# In[ ]:


#Accuracy is very low


# In[ ]:




