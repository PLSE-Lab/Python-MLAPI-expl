#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


data=pd.read_csv('../input/advertising.csv')
data.head()


# In[3]:


data.columns=['Daily_Time_Spent_on_Site', 'Age', 'Area Income',
       'Daily_Internet_Usage', 'Ad_Topic_Line', 'City', 'Male', 'Country',
       'Timestamp', 'Clicked_on_Ad']


# In[4]:


data.drop(['Ad_Topic_Line','City','Timestamp'],axis=1,inplace=True)


# In[5]:


data.drop(['Country'],axis=1,inplace=True)


# In[6]:


data.tail()


# In[13]:


data.info()


# In[14]:


data.isnull().sum()


# In[7]:


x=data.Clicked_on_Ad
y=data.drop(['Clicked_on_Ad'],axis=1)


# In[8]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(y,x,test_size=0.2,random_state=42)


# In[9]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[11]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
print('test accuracy {}'.format(lr.score(x_train,y_train)))


# In[12]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,lr.predict(y_test))
print(cm)

