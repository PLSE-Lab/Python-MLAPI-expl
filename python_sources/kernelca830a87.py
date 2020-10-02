#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv("../input/HeartDisease.csv")


# In[ ]:


data.head()


# In[ ]:


# Get the number of missing data points per column. This will show up in variable explorer
missing_values_count = data.isnull().sum()
print(missing_values_count)


# In[ ]:



#Taking care of missing values
data = data.dropna()


# In[ ]:


data.tail()


# In[ ]:


df=data.groupby('Sex').mean()
df


# In[ ]:


b=df.drop(['ID','Age','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak'],axis=1)
b


# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


plt.figure(figsize=(16, 8))
b.plot(kind='bar')


# In[ ]:


c=df.drop(['ID','Age','cp','trestbps','fbs','restecg','thalach','exang','oldpeak','num'],axis=1)
c


# In[ ]:


plt.figure(figsize=(10,5))
c.plot(kind='bar')


# In[ ]:


d=df.drop(['ID','Age','cp','trestbps','chol','restecg','thalach','exang','oldpeak','num'],axis=1)
d


# In[ ]:


plt.figure(figsize=(10,5))
d.plot(kind='bar')


# In[ ]:


data.describe()


# In[ ]:


#Plot only the values of num- the value to be predicted/Label
data["num"].value_counts().sort_index().plot.bar()


# In[ ]:


a=data['Age'].value_counts()
a


# In[ ]:


plt.figure(figsize=(20,10))
a.plot(kind='bar')


# In[ ]:


import seaborn as sb


# In[ ]:


plt.figure(figsize=(20,10))
sb.barplot(x='Age', y='chol', data=data)


# In[ ]:


plt.figure(figsize=(20,10))
sb.barplot(x='Age', y='num', data=data)


# In[ ]:


#Heat map to see the coreelation between variables, use annot if you want to see the values in the heatmap
plt.subplots(figsize=(12,8))
sb.heatmap(data.corr(),robust=True,annot=True)


# In[ ]:


#Detect outliers
plt.subplots(figsize=(15,6))
data.boxplot(patch_artist=True, sym="k.")


# In[ ]:


data.info()


# In[ ]:


data.columns


# In[ ]:


#linear model
X=data[['chol', 'Age', 'cp', 'fbs']]
y=data['num']


# In[ ]:


#training set
from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
X_test.tail()
#random_state-> same random splits


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


reg=LinearRegression()


# In[ ]:


reg.fit(X_train,y_train)


# In[ ]:


#coefficient
print(reg.intercept_)


# In[ ]:


reg.coef_


# In[ ]:


X_train.columns


# In[ ]:


m=pd.DataFrame(reg.coef_,X.columns, columns=['Coeff'])
m


# In[ ]:


predictions=reg.predict(X_test)


# In[ ]:


predictions


# In[ ]:


y_test.head()


# In[ ]:


plt.scatter(y_test, predictions)


# In[ ]:


reg.score(X_test,y_test)


# In[ ]:





# In[ ]:




