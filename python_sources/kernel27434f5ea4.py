#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv('/kaggle/input/80-cereals/cereal.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# There are no null values

# In[ ]:


numerical=df.select_dtypes(include=['int64','float64'])
numerical.head()


# In[ ]:


categorical=df.select_dtypes(include='object')
categorical.head()


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.distplot(df['rating'])


# In[ ]:


sns.barplot(data=df,x='rating',y='sugars')


# As the value of sugar increases in the ceareals the rating of the cereals decreases

# In[ ]:


df.columns


# In[ ]:


sns.barplot(data=df,x='protein',y='rating')


# As the protien content in the cereals increase then the rating for the cereals also increases

# In[ ]:


sns.barplot(data=df,x='vitamins',y='rating')


# when the vitamin is 0 the rating of the cereal is very high whereas the rating is low for the high vitamin cereals 

# In[ ]:


sns.barplot(df['fat'],df['rating'])


# when the fat content in the cereals is 0 rating for the cereals is high while the fat content in the cereal decreses the rating of the cereals.
# when fat content is 5 then rating is very low

# In[ ]:


df.columns


# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot(df['sodium'],df['rating'])


# sodium content should be between 0-130 for better rating

# In[ ]:


sns.barplot(df['fiber'],df['rating'])


# increased fiber content increased rating

# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot(df['carbo'],df['rating'])


# In[ ]:


corr=df.corr()


# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(corr, cbar = True,  square = True, annot=True,cmap= 'coolwarm')


# In[ ]:


categorical.head()


# In[ ]:


dummy=pd.get_dummies(df[['name','mfr','type']])
column_name=df.columns.values.tolist() 
column_name.remove('name') 
column_name.remove('mfr')
column_name.remove('type')
data1=df[column_name].join(dummy) 


# In[ ]:


data1.head()


# In[ ]:


from sklearn.model_selection import train_test_split
y = data1['rating']
X = data1.drop('rating', axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=27)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
import statsmodels.api as sm
lm = sm.OLS(y_train, X_train).fit()
lm.summary()


# In[ ]:




