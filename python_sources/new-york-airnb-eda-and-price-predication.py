#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(df['neighbourhood_group'])


# In[ ]:


df.fillna('last_review',inplace=True)


# In[ ]:


df.info()


# In[ ]:


sns.countplot(df['room_type'])


# In[ ]:


pd.crosstab(df['room_type'],df['neighbourhood_group']).plot.bar(figsize=(15,8))


# In[ ]:


brooynl = df[df['neighbourhood_group']=="Brooklyn"][['neighbourhood','price']].groupby(['neighbourhood']).mean()
sns.distplot(brooynl)


# In[ ]:


private_room_price = df[df.room_type=="Private room"][['neighbourhood_group','price']].groupby('neighbourhood_group')['price'].mean()
sns.distplot(private_room_price)


# In[ ]:


Entireapt = df[df.room_type=="Entire home/apt"][['neighbourhood_group','price']].groupby('neighbourhood_group')['price'].mean()
sns.distplot(Entireapt)


# In[ ]:


shared_room = df[df.room_type=="Shared room"][['neighbourhood_group','price']].groupby('neighbourhood_group')['price'].mean()
sns.distplot(shared_room)


# In[ ]:


plt.figure(figsize=(15,7))
sns.distplot(df['availability_365'])


# In[ ]:


shared_mimum_night = df[df.room_type=="Shared room"]['minimum_nights']
plt.figure(figsize=(15,7))
plt.xlabel("minum Nights")
sns.swarmplot(y= shared_mimum_night.index,x= shared_mimum_night.values)


# In[ ]:


plt.figure(figsize=(15,7))
sns.scatterplot(y=df.latitude,x=df.longitude,hue=df.availability_365)


# In[ ]:


plt.figure(figsize=(15,7))
sns.scatterplot(y=df.latitude,x=df.longitude,hue=df.neighbourhood_group)


# In[ ]:


most_popular = (df.host_id.value_counts()[:15])
plt.figure(figsize=(15,7))
sns.barplot(y=most_popular.values,x=most_popular.index)


# In[ ]:


enc = LabelEncoder()
df['name'] = enc.fit_transform(df['name'])
df['neighbourhood'] = enc.fit_transform(df['neighbourhood'])
df['neighbourhood_group'] = enc.fit_transform(df['neighbourhood_group'])
df['room_type'] = enc.fit_transform(df['room_type'])


# In[ ]:


df.head()


# In[ ]:


x = df.drop(['price','host_name','latitude','longitude','number_of_reviews','last_review','reviews_per_month'],axis=1)
y = df['price']
# standScalar = StandardScaler()
# x = standScalar.fit_transform(x)
#Getting Test and Training Set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=353)
x_train.head()
y_train.head()


# In[ ]:


reg=DecisionTreeClassifier()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
from sklearn.metrics import r2_score,accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:


reg=ExtraTreesClassifier()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
accuracy_score(y_test,y_pred)


# In[ ]:


reg=NearestNeighbors()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
from sklearn.metrics import r2_score,accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:




